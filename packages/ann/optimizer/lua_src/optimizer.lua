local math = math
local table = table
local string = string
--
local ipairs = ipairs
local pairs = pairs
local assert = assert
--
local type = type
local iterator = iterator
local get_table_fields = get_table_fields
local april_assert = april_assert

------------------------------------------------------------------------------

local MAX_UPDATES_WITHOUT_PRUNE=100

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
-- REMOVE UTILS FROM GLOBALS TABLE
local ann_optimizer_utils = ann.optimizer.utils
ann.optimizer.utils = nil

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

get_table_from_dotted_string("ann.optimizer.regularizations", true)

function ann.optimizer.regularizations.weight_decay(dest, wd, w)
  if wd > 0.0 then
    dest:axpy(-wd, w)
  end
end

-- This regularization term must be applied the last
function ann.optimizer.regularizations.L1_norm(dest, value, w)
  if value > 0.0 then
    ann_optimizer_utils.regularization.L1_norm_map(dest, value, w)
  end
end

------------------------------------------------------------------------------

get_table_from_dotted_string("ann.optimizer.constraints", true)

-- The penalty is computed and applied on w
function ann.optimizer.constraints.max_norm_penalty(mp, w)
  if mp > 0.0 then
    local sw         = w:sliding_window()
    local window     = nil
    while not sw:is_end() do
      window  = sw:get_matrix(window)
      local norm2 = window:norm2()
      if norm2 > mp then
	local scal_factor = mp / norm2
	window:scal(scal_factor)
      end
      sw:next()
    end
  end
end

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- global environment ann.optimizer
local optimizer,optimizer_methods = class("ann.optimizer", nil, ann.optimizer)

function optimizer:constructor(valid_options,
                               g_options,
                               l_options,
                               count)
  local g_options, l_options = g_options or {}, l_options or {}
  self.valid_options     = iterator(ipairs(valid_options or {})):map(function(i,t)return t[1],t[2] end):table()
  self.global_options    = {}
  self.layerwise_options = {}
  self.count             = count or 0
  self.regularizations   = {}
  self.constraints       = {}
  self.regularizations_order = {}
  self.constraints_order     = {}
  for name,value in pairs(g_options) do
    self.global_options[name] = value
  end
  for layer_name,options in pairs(l_options) do
    for name,value in pairs(options) do
      self.layerwise_options[layer_name] = self.layerwise_options[layer_name] or {}
      self.layerwise_options[layer_name][name] = value
    end
  end
  return obj
end


-- regularization functions has the API: func(dest, value, w) where dest is the
-- destination matrix, and w the current weights matrix
function optimizer_methods:add_regularization(hyperparameter_name, func, desc)
  local func = april_assert(func or ann.optimizer.regularizations[hyperparameter_name],
			    "Problem with hyperparemter function of %s",
			    hyperparameter_name)
  april_assert(not self.valid_options[hyperparameter_name],
	       "Redefinition of hyperparameter %s",
	       hyperparameter_name)
  self.valid_options[hyperparameter_name] = desc
  self.regularizations[hyperparameter_name] = func
  table.insert(self.regularizations_order, hyperparameter_name)
end

-- constraint functions has the API: func(value, w)
function optimizer_methods:add_constraint(hyperparameter_name, func, desc)
  local func = april_assert(func or ann.optimizer.constraints[hyperparameter_name],
			    "Problem with hyperparemter function of %s",
			    hyperparameter_name)
  april_assert(not self.valid_options[hyperparameter_name],
	       "Redefinition of hyperparameter %s",
	       hyperparameter_name)
  self.valid_options[hyperparameter_name] = desc
  self.constraints[hyperparameter_name] = func
  table.insert(self.constraints_order, hyperparameter_name)
end

local function ann_optimizer_apply_regularizations(opt, wname, dest, w)
  for _,hypname in ipairs(opt.regularizations_order) do
    local func = opt.regularizations[hypname]
    local v = opt:get_option_of(wname, hypname)
    if v then
      -- sanity check
      if v > 0.0 and #w:dim() == 2 and w:dim(2) == 1 then
	fprintf(io.stderr,
		"# WARNING!!! Possible %s > 0 in bias connection: %s\n",
		hypname, wname)
      end
      func(dest, v, w)
    end
  end
end

local function ann_optimizer_apply_constraints(opt, wname, w)
  for _,hypname in pairs(opt.constraints_order) do
    local func = opt.constraints[hypname]
    local v = opt:get_option_of(wname, hypname)
    if v then
      -- sanity check
      if v > 0.0 and #w:dim() == 2 and w:dim(2) == 1 then
	fprintf(io.stderr,
		"# WARNING!!! Possible %s > 0 in bias connection: %s\n",
		hypname, wname)
      end
      func(v, w)
    end
  end
end

function optimizer_methods:show_options()
  local t = iterator(pairs(self.valid_options)):enumerate():table()
  table.sort(t, function(a,b) return a[1]<b[1] end)
  print(iterator(ipairs(t)):select(2):map(table.unpack):concat("\t","\n"))
end

function optimizer_methods:has_option(name)
  return self.valid_options[name]
end

function optimizer_methods:set_option(name,value)
  april_assert(self.valid_options[name], "Not recognized option %s", name)
  self.global_options[name] = value
  return self
end

function optimizer_methods:get_option(name)
  april_assert(self.valid_options[name], "Not recognized option %s", name)
  return self.global_options[name]
end

function optimizer_methods:set_layerwise_option(layer_name,name,value)
  april_assert(self.valid_options[name], "Not recognized option %s", name)
  self.layerwise_options[layer_name] = self.layerwise_options[layer_name] or {}
  self.layerwise_options[layer_name][name] = value
  return self
end

function optimizer_methods:get_layerwise_option(layer_name,name)
  april_assert(self.valid_options[name], "Not recognized option %s", name)
  return (self.layerwise_options[layer_name] or {})[name]
end

function optimizer_methods:get_option_of(layer_name,name)
  april_assert(self.valid_options[name], "Not recognized option %s", name)
  return ( (self.layerwise_options[layer_name] or {})[name] or
	     self.global_options[name] )
end

-- eval is a function which returns the data needed by the optimizer (at least,
-- the loss, and the gradients. The rest of values will be ignored)
--
-- weights is a dictionary of weight matrix objects, indexed by its names, or a
-- matrix
function optimizer_methods:execute(eval, weights)
  error("NOT IMPLEMENTED METHOD!, use a derived class instance")
end

function optimizer_methods:count_one()
  self.count = self.count + 1
end

function optimizer_methods:get_count()
  return self.count
end

function optimizer_methods:clone()
  local obj = ann.optimizer()
  obj.count = self.count
  return obj
end

function optimizer_methods:needs_property(name)
  return false
end

------------------------------------------------
------------------------------------------------
------------------------------------------------

local function ann_optimizer_apply_momentum(mt, update)
  if mt > 0.0 then
    -- intertia is computed as a portion of previous update
    update:scal(mt)
  else
    -- sets to ZERO
    update:zeros()
  end
end

------------------------------------------------
------------------------------------------------
------------------------------------------------

local wrap_matrices = matrix.dict.wrap_matrices

------------------------------------------------
--------- STOCHASTIC GRADIENT DESCENT ----------
------------------------------------------------

local sgd, sgd_methods = class("ann.optimizer.sgd", ann.optimizer)
ann.optimizer.sgd = sgd -- global environment

function sgd:constructor(g_options, l_options, count, update)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"learning_rate", "Learning speed factor (0.1)"},
			      {"momentum", "Learning inertia factor (0.1)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.update = wrap_matrices(update or matrix.dict())
  -- standard regularization and constraints
  self:add_regularization("weight_decay", nil, "Weight L2 regularization (1e-04)")
  self:add_regularization("L1_norm", nil, "Weight L1 regularization (1e-05)")
  self:add_constraint("max_norm_penalty", nil, "Weight max norm upper bound (4)")
end

function sgd_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  --
  local origw = weights
  local weights = wrap_matrices(weights)
  local arg = table.pack( eval(origw) )
  local tr_loss,gradients = table.unpack(arg)
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  gradients = wrap_matrices(gradients)
  for name,w in pairs(weights) do
    local update      = self.update(name) or w:clone():zeros()
    local grad        = gradients(name)
    local lr          = assert(self:get_option_of(name, "learning_rate"),
			       "The learning_rate parameter needs to be set")
    local mt          = self:get_option_of(name, "momentum") or 0.0
    --
    ann_optimizer_apply_momentum(mt, update)
    -- apply back-propagation learning rule
    update:axpy(-lr, grad)
    -- regularizations
    ann_optimizer_apply_regularizations(self, name, update, w)
    -- apply update matrix to the weights
    w:axpy(1.0, update)
    -- constraints
    ann_optimizer_apply_constraints(self, name, w)
    --
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
    --
    self.update[name] = update
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function sgd_methods:clone()
  local obj = ann.optimizer.sgd()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.update            = self.update:clone()
  return obj
end

function sgd_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.sgd(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  self.update:to_lua_string(format),
		  ")" }
  return table.concat(str_t, "")
end

local sgd_properties = {
  gradient = true
}
function sgd_methods:needs_property(name)
  return sgd_properties[name]
end

-----------------------------------
--------- RESILIENT PROP ----------
-----------------------------------

local rprop, rprop_methods = class("ann.optimizer.rprop", ann.optimizer)
ann.optimizer.rprop = rprop

function rprop:constructor(g_options, l_options, count,
                           steps, old_sign)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"initial_step", "Initial weight update value (0.1)"},
			      {"eta_plus", "Update value up by this factor (1.2)"},
			      {"eta_minus", "Update value down by this factor (0.5)"},
			      {"max_step", "Maximum value of update step (50)"},
			      {"min_step", "Minimum value of update step (1e-05)"},
			      {"niter", "Number of iterations (1)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.steps    = wrap_matrices(steps or matrix.dict())
  self.old_sign = wrap_matrices(old_sign or matrix.dict())
  self:set_option("initial_step",  0.1)
  self:set_option("eta_plus",      1.2)
  self:set_option("eta_minus",     0.5)
  self:set_option("max_step",      50)
  self:set_option("min_step",      1e-05)
  self:set_option("niter",         1)
end

function rprop_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  --
  local origw = weights
  local weights = wrap_matrices(weights)
  local initial_step  = self:get_option("initial_step")
  local eta_plus      = self:get_option("eta_plus")
  local eta_minus     = self:get_option("eta_minus")
  local max_step      = self:get_option("max_step")
  local min_step      = self:get_option("min_step")
  local niter         = self:get_option("niter")
  local steps         = self.steps
  local old_sign      = self.old_sign
  local arg
  for i=1,niter do
    arg = table.pack( eval(origw, i-1) )
    local tr_loss,gradients = table.unpack(arg)
    -- the gradient computation could fail returning nil, it is important to
    -- take this into account
    if not gradients then return nil end
    gradients = wrap_matrices(gradients)
    --
    for name,w in pairs(weights) do
      steps[name] = steps(name) or w:clone():fill(initial_step)
      local sign  = gradients(name):clone():sign()
      -- apply reprop learning rule
      if old_sign(name) then
	ann_optimizer_utils.rprop.step(steps(name),
				       old_sign(name),
				       sign,
				       eta_minus,
				       eta_plus)
      end
      steps(name):clamp(min_step, max_step)
      w:axpy(-1.0, sign:clone():cmul(steps(name)))
      -- keep the sign for the next iteration
      old_sign[name] = sign
      --
      if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
	w:prune_subnormal_and_check_normal()
      end
    end
    -- count one more update iteration
    self:count_one()
  end
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function rprop_methods:clone()
  local obj = ann.optimizer.rprop()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  if self.steps then
    obj.steps = self.steps:clone()
  end
  if self.old_sign then
    obj.old_sign = self.old_sign:clone()
  end
  return obj
end

function rprop_methods:to_lua_string(format)
  local str_t = { "ann.optimizer.rprop(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  self.steps:to_lua_string(format),
		  ",",
		  self.old_sign:to_lua_string(format),
		  ")" }
  return table.concat(str_t, "")
end

local rprop_properties = {
  gradient = true
}
function rprop_methods:needs_property(name)
  return rprop_properties[name]
end

---------------------------------------
--------- CONJUGATE GRADIENT ----------
---------------------------------------

-- Conjugate Gradient implementation, copied/modified from optim package of
-- Torch 7, which is a rewrite of minimize.m written by Carl E. Rasmussen.
local cg, cg_methods = class("ann.optimizer.cg", ann.optimizer)
ann.optimizer.cg = cg

function cg:constructor(g_options, l_options, count,
                        df0, df1, df2, df3, x0, s)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
                              --			      {"momentum", "Learning inertia factor (0.1)"},
                              {"rho", "Constant for Wolf-Powell conditions (0.01)"},
                              {"sig", "Constant for Wolf-Powell conditions (0.5)"},
                              {"int", "Reevaluation limit (0.1)"},
                              {"ext", "Maximum number of extrapolations (3)"},
                              {"max_iter", "Maximum number of iterations (20)"},
                              {"ratio", "Maximum slope ratio (100)"},
                              {"max_eval", "Maximum number of evaluations (max_iter*1.25)"},
                            },
                            g_options,
                            l_options,
                            count)
  self.state    = {
    df0 = df0,
    df1 = df1,
    df2 = df2, 
    df3 = df3,
    x0  = x0,
    s   = s,
  }
  self:set_option("rho",           0.01) -- rho is a constant in Wolfe-Powell conditions
  self:set_option("sig",           0.5)  -- sig is another constant of Wolf-Powell
  self:set_option("int",           0.1)  -- reevaluation limit
  self:set_option("ext",           3.0)  -- maximum number of extrapolations
  self:set_option("max_iter",      20)
  self:set_option("ratio",         100)  -- maximum slope ratio
  -- standard regularization and constraints
  self:add_regularization("weight_decay", nil, "Weight L2 regularization (1e-04)")
  self:add_regularization("L1_norm", nil, "Weight L1 regularization (1e-05)")
  self:add_constraint("max_norm_penalty", nil, "Weight max norm upper bound (4)")
end

function cg_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  local math = math
  --
  local origw = weights
  local weights = wrap_matrices(weights)
  -- UPDATE_WEIGHTS function
  local update_weights = function(x, dir, s)
    x:axpy(dir, s)
  end
  -- APPLY REGULARIZATION AND PENALTIES
  local apply_regularization_and_penalties = function(x)
    for name,w in pairs(x) do
      -- regularizations
      ann_optimizer_apply_regularizations(self, name, w, w)
      -- constraints
      ann_optimizer_apply_constraints(self, name, w)
    end
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      x:prune_subnormal_and_check_normal()
    end
  end
  ----------------------------------------------------------------------------
  
  -- count one more update iteration
  self:count_one()
  
  local x             = weights
  local rho           = self:get_option("rho")
  local sig           = self:get_option("sig")
  local int           = self:get_option("int")
  local ext           = self:get_option("ext")
  local max_iter      = self:get_option("max_iter")
  local ratio         = self:get_option("ratio")
  local max_eval      = self:get_option("max_eval") or max_iter*1.25
  local red           = 1
  
  local i             = 0 -- counts the number of evaluations
  local ls_failed     = 0
  local fx            = {}

  -- we need three points for the interpolation/extrapolation stuff
  local z1,z2,z3 = 0,0,0
  local d1,d2,d3 = 0,0,0
  local f1,f2,f3 = 0,0,0

  local df1 = self.state.df1 or x:clone_only_dims()
  local df2 = self.state.df2 or x:clone_only_dims()
  local df3 = self.state.df3 or x:clone_only_dims()
  
  -- search direction
  local s = self.state.s or x:clone_only_dims()
  
  -- we need a temp storage for X
  local x0  = self.state.x0 or x:clone()
  local f0  = 0
  local df0 = self.state.df0 or x:clone_only_dims()
  
  -- evaluate at initial point
  local arg = table.pack( eval(origw, i) )
  local tr_loss,gradients = table.unpack(arg)
  if not gradients then return nil end
  gradients = wrap_matrices(gradients)
  f1 = tr_loss
  table.insert(fx, f1)
  df1:copy(gradients)
  i=i+1
  
  -- initial search direction
  s:copy(df1)
  s:scal(-1)
  
  -- slope
  d1 = -s:dot(s)
  -- initial step
  z1 = red/(1-d1)
  
  while i < math.abs(max_eval) do
    
    x0:copy(x)
    
    f0 = f1
    df0:copy(df1)
    
    update_weights(x, z1, s)

    arg = table.pack( eval(origw, i) )
    tr_loss,gradients = table.unpack(arg)
    gradients = wrap_matrices(gradients)
    f2 = tr_loss
    
    df2:copy(gradients)
    i=i+1
    d2 = df2:dot(s)
    -- init point 3 equal to point 1
    f3,d3,z3 = f1,d1,-z1
    local m       = math.min(max_iter,max_eval-i)
    local success = false
    local limit   = -1
    
    while true do
      while (f2 > f1+z1*rho*d1 or d2 > -sig*d1) and m > 0 do
	limit = z1
	if f2 > f1 then
	  z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)
	else
	  local A = 6*(f2-f3)/z3+3*(d2+d3)
	  local B = 3*(f3-f2)-z3*(d3+2*d2)
	  z2 = (math.sqrt(B*B-A*d2*z3*z3)-B)/A
	end
	if z2 ~= z2 or z2 == math.huge or z2 == -math.huge then
	  z2 = z3/2
	end
	z2 = math.max(math.min(z2, int*z3),(1-int)*z3)
	z1 = z1 + z2
	
	update_weights(x, z2, s)
	arg = table.pack( eval(origw, i) )
	tr_loss,gradients = table.unpack(arg)
	gradients = wrap_matrices(gradients)
	f2 = tr_loss
	df2:copy(gradients)
	i=i+1
	m = m - 1
	d2 = df2:dot(s)
	z3 = z3-z2
      end
      if f2 > f1+z1*rho*d1 or d2 > -sig*d1 then
	break
      elseif d2 > sig*d1 then
	success = true
	break
      elseif m == 0 then
	break
      end
      local A = 6*(f2-f3)/z3+3*(d2+d3);
      local B = 3*(f3-f2)-z3*(d3+2*d2);
      z2 = -d2*z3*z3/(B+math.sqrt(B*B-A*d2*z3*z3))
      
      if z2 ~= z2 or z2 == math.huge or z2 == -math.huge or z2 < 0 then
	if limit < -0.5 then
	  z2 = z1 * (ext -1)
	else
	  z2 = (limit-z1)/2
	end
      elseif (limit > -0.5) and (z2+z1) > limit then
	z2 = (limit-z1)/2
      elseif limit < -0.5 and (z2+z1) > z1*ext then
	z2 = z1*(ext-1)
      elseif z2 < -z3*int then
	z2 = -z3*int
      elseif limit > -0.5 and z2 < (limit-z1)*(1-int) then
	z2 = (limit-z1)*(1-int)
      end
      f3=f2
      d3=d2
      z3=-z2
      z1=z1+z2;
      update_weights(x, z2, s)
      
      arg = table.pack( eval(origw, i) )
      tr_loss,gradients = table.unpack(arg)
      gradients = wrap_matrices(gradients)
      f2 = tr_loss
      df2:copy(gradients)
      i=i+1
      m = m - 1
      d2 = df2:dot(s)
    end
    if success then
      f1 = f2
      table.insert(fx, f1)
      local ss = (df2:dot(df2) - df2:dot(df1))/df1:dot(df1)
      s:scal(ss)
      s:axpy(-1,df2)
      df1,df2 = df2,df1
      -- local tmp = clone(df1)
      -- copy(df1,df2)
      -- copy(df2,tmp)
      d2 = df1:dot(s)
      if d2> 0 then
	s:copy(df1)
	s:scal(-1)
	d2 = -s:dot(s)
      end
      z1 = z1 * math.min(ratio, d1/(d2-1e-320))
      d1 = d2
      ls_failed = 0
    else
      x:copy(x0)
      f1 = f0
      df1:copy(df0)
      if ls_failed or i>max_eval then
	break
      end
      df1,df2 = df2,df1
      -- local tmp = clone(df1)
      -- copy(df1,df2)
      -- copy(df2,tmp)
      s:copy(df1)
      s:scal(-1)
      d1 = -s:dot(s)
      z1 = 1/(1-d1)
      ls_failed = 1
    end
  end
  self.state.df0 = df0
  self.state.df1 = df1
  self.state.df2 = df2
  self.state.df3 = df3
  self.state.x0 = x0
  self.state.s = s
  
  apply_regularization_and_penalties(x)
  
  -- evaluate the function at the end
  local arg = table.pack( eval(origw, i) )
  -- returns the same as returned by eval(), plus the sequence of iteration
  -- losses and the number of iterations
  table.insert(arg, fx)
  table.insert(arg, i)
  return table.unpack(arg)
end

function cg_methods:clone()
  local obj = ann.optimizer.cg()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  if self.state.df0 then
    obj.state.df0 = self.state.df0:clone()
  end
  if self.state.df1 then
    obj.state.df1 = self.state.df1:clone()
  end
  if self.state.df2 then
    obj.state.df2 = self.state.df2:clone()
  end
  if self.state.df3 then
    obj.state.df3 = self.state.df3:clone()
  end
  if self.state.x0 then
    obj.state.x0 = self.state.x0:clone()
  end
  if self.state.s then
    obj.state.s = self.state.s:clone()
  end
  return obj
end

function cg_methods:to_lua_string(format)
  local str_t = { "ann.optimizer.cg(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ")" }
  return table.concat(str_t, "")
end

local cg_properties = {
  gradient = true
}
function cg_methods:needs_property(name)
  return cg_properties[name]
end

------------------------------
--------- QUICKPROP ----------
------------------------------

local quickprop, quickprop_methods = class("ann.optimizer.quickprop",
                                           ann.optimizer)
ann.optimizer.quickprop = quickprop

function quickprop:constructor(g_options, l_options, count,
                               update, lastg)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
                              {"learning_rate", "Learning speed factor (0.1)"},
                              {"mu", "Maximum growth factor (1.75)"},
                              {"epsilon", "Bootstrap factor (1e-04)"},
			      {"max_step", "Maximum step value (1000)"},
			    },
			    g_options,
			    l_options,
			    count)
  self:set_option("mu", 1.75)
  self:set_option("epsilon", 1e-04)
  self:set_option("max_step", 1000)
  self.update = wrap_matrices(update or matrix.dict())
  self.lastg  = wrap_matrices(lastg  or matrix.dict())
  -- standard regularization and constraints
  self:add_regularization("weight_decay", nil, "Weight L2 regularization (1e-04)")
  self:add_regularization("L1_norm", nil, "Weight L1 regularization (1e-05)")
  self:add_constraint("max_norm_penalty", nil, "Weight max norm upper bound (4)")
end

function quickprop_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  local math = math
  --
  local origw = weights
  local weights = wrap_matrices(weights)
  local arg = table.pack( eval(origw) )
  local tr_loss,gradients = table.unpack(arg)
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  gradients = wrap_matrices(gradients)
  for name,w in pairs(weights) do
    local update      = self.update(name)
    local lastg       = self.lastg(name)
    local grad        = gradients(name)
    local lr          = assert(self:get_option_of(name, "learning_rate"),
			       "The learning_rate parameter needs to be set")
    local mu          = self:get_option_of(name, "mu")
    local epsilon     = self:get_option_of(name, "epsilon")
    local max_step    = self:get_option_of(name, "max_step")
    if not update then
      -- compute standard back-propagation learning rule
      update = w:clone()
      lastg  = grad:clone()
      update:copy(grad)
    else
      local shrink = mu / (1.0 + mu)
      -- compute quickprop update
      update:map(lastg, grad,
		 function(prev_step, prev_slope, slope)
		   local step=0
		   if math.abs(prev_step) > 1e-03 then
		     if math.sign(slope) == math.sign(prev_step) then
		       step = step + epsilon * slope
		     end
		     if ( (prev_step > 0 and slope > shrink*prev_slope) or
		     	  (prev_step < 0 and slope < shrink*prev_slope) ) then
		       step = step + mu * prev_step
		     else
		       step = step + (prev_step*slope) / (prev_slope - slope)
		     end
		   else
		     step = step + epsilon * slope
		   end
		   if step > max_step then step = max_step
		   elseif step < -max_step then
		     step = -max_step
		   end
		   return step
		 end)
      lastg:copy(grad)
    end
    --
    self.update[name] = update
    self.lastg[name]  = lastg
    -- regularizations
    ann_optimizer_apply_regularizations(self, name, update, w)
    -- apply update matrix to the weights
    w:axpy(-lr, update)
    -- constraints
    ann_optimizer_apply_constraints(self, name, w)
    --
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function quickprop_methods:clone()
  local obj = ann.optimizer.quickprop()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.update            = self.update:clone()
  obj.lastg             = self.lastg:clone()
  return obj
end

function quickprop_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.quickprop(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  self.update:to_lua_string(format),
		  ",",
		  self.lastg:to_lua_string(format),
		  ")" }
  return table.concat(str_t, "")
end

local quickprop_properties = {
  gradient = true
}
function quickprop_methods:needs_property(name)
  return quickprop_properties[name]
end


---------------------------------------------------------
--------- AVERAGED STOCHASTIC GRADIENT DESCENT ----------
---------------------------------------------------------

-- extracted from: http://research.microsoft.com/pubs/192769/tricks-2012.pdf
-- Leon Bottou, Stochastic Gradient Descent Tricks, Microsoft Research, 2012

local asgd, asgd_methods = class("ann.optimizer.asgd", ann.optimizer)
ann.optimizer.asgd = asgd

function asgd:constructor(g_options, l_options, count, update)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"learning_rate", "Learning speed factor (0.1)"},
			      {"lr_decay", "Learning decay factor (0.75)"},
			      {"t0", "Average starts at bunch t0, good values are data size or data dimension (0)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.update = wrap_matrices(update or matrix.dict())
  -- standard regularization and constraints
  self:add_regularization("weight_decay", nil, "Weight L2 regularization (1e-04)")
  -- default values
  self:set_option("lr_decay", 0.75)
  self:set_option("t0", 0)
end

function asgd_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  --
  local origw = weights
  local weights = wrap_matrices(weights)
  local arg = table.pack( eval(origw) )
  local tr_loss,gradients = table.unpack(arg)
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  gradients = wrap_matrices(gradients)
  local t = self:get_count()
  for name,w in pairs(weights) do
    local update      = (self.update(name) or w:clone()):zeros()
    local grad        = gradients(name)
    local lr          = assert(self:get_option_of(name, "learning_rate"),
			       "The learning_rate parameter needs to be set")
    local lr_decay    = self:get_option_of(name, "lr_decay")
    local t0          = self:get_option_of(name, "t0")
    -- effective values at time t
    local lr_t        = lr / ((1.0 + lr * t)^(lr_decay)) -- learning rate factor
    local mu_t        = 1.0 / math.max(1, t - t0)        -- average factor
    -- apply back-propagation learning rule
    update:axpy(-lr, grad)
    -- regularizations
    ann_optimizer_apply_regularizations(self, name, update, w)
    -- compute averaged weights
    if mu_t > 1 or mu_t < 1 then
      update:axpy(1.0,w)
      w:scal(1.0 - mu_t):axpy(mu_t, update)
    else
      w:axpy(1.0, update)
    end
    --
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
    --
    self.update[name] = update
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function asgd_methods:clone()
  local obj = ann.optimizer.asgd()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.update            = self.update:clone()
  return obj
end

function asgd_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.asgd(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  self.update:to_lua_string(format),
		  ")" }
  return table.concat(str_t, "")
end

local asgd_properties = {
  gradient = true
}
function asgd_methods:needs_property(name)
  return asgd_properties[name]
end
