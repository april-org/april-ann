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

local function ann_optimizer_regularizations_weight_decay(destw, cwd, w)
  -- sum current weights by applying the complementary of weight decay term. We
  -- derive w, and left the result on destw
  destw:axpy(cwd, w)  
end

------------------------------------------------------------------------------

get_table_from_dotted_string("ann.optimizer.regularizations", true)

function ann.optimizer.regularizations.L1_norm(destw, value, w)
  if value > 0.0 then
    ann_optimizer_utils.regularization.L1_norm_map(destw, value, w)
  end
end

------------------------------------------------------------------------------

get_table_from_dotted_string("ann.optimizer.constraints", true)

-- The penalty is computed on w, but applied to oldw and w
function ann.optimizer.constraints.max_norm_penalty(w, oldw, mp)
  if mp > 0.0 then
    local sw         = w:sliding_window()
    local window     = nil
    local old_sw     = oldw:sliding_window()
    local old_window = nil
    while not sw:is_end() do
      window  = sw:get_matrix(window)
      local norm2 = window:norm2()
      if norm2 > mp then
	local scal_factor = mp / norm2
	old_window = old_sw:get_matrix(old_window)
	old_window:scal(scal_factor)
	window:scal(scal_factor)
      end
      old_sw:next()
      sw:next()
    end
  end
end

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local optimizer_methods,
optimizer_class_metatable = class("ann.optimizer")

function optimizer_class_metatable:__call(valid_options,
					  g_options,
					  l_options,
					  count)
  local g_options, l_options = g_options or {}, l_options or {}
  local obj = class_instance({
			       valid_options     = iterator(ipairs(valid_options or {})):map(function(i,t)return t[1],t[2] end):table(),
			       global_options    = {},
			       layerwise_options = {},
			       count             = count or 0,
			       regularizations   = {},
			       constraints       = {},
			     },
			     self)
  for name,value in pairs(g_options) do obj:set_option(name,value) end
  for wname,options in pairs(l_options) do
    for name,value in pairs(options) do
      obj:set_layerwise_option(wname,name,value)
    end
  end
  return obj
end


-- regularization functions has the API:
-- func(oldw, value, w, ann_component)
-- where oldw is the destination weights matrix, and w the current weights matrix
function optimizer_methods:add_regularization(hyperparameter_name, func, desc)
  local func = assert(func or ann.optimizer.regularizations[hyperparameter_name],
		      "Problem with hyperparemter function of " .. hyperparameter_name)
  assert(not self.valid_options[hyperparameter_name],
	 "Redefinition of hyperparameter " .. hyperparameter_name)
  self.valid_options[hyperparameter_name] = desc
  self.regularizations[hyperparameter_name] = func
end

-- constraint functions has the API:
-- func(oldw, value, w, ann_component) => remember to apply the same constraint to oldw and w
-- where oldw is the next weights matrix, and w the current weights matrix
function optimizer_methods:add_constraint(hyperparameter_name, func, desc)
  local func = assert(func or ann.optimizer.constraints[hyperparameter_name],
		      "Problem with hyperparemter function of " .. hyperparameter_name)
  assert(not self.valid_options[hyperparameter_name],
	 "Redefinition of hyperparameter " .. hyperparameter_name)
  self.valid_options[hyperparameter_name] = desc
  self.constraints[hyperparameter_name] = func
end

local function ann_optimizer_apply_regularizations(opt,
						   wname, oldw, w,
						   ann_component)
  for hypname,func in pairs(opt.regularizations) do
    local v = opt:get_option_of(wname, hypname)
    if v then
      -- sanity check
      if v > 0.0 and w:dim(2) == 1 then
	fprintf(io.stderr,
		"# WARNING!!! Possible " .. hypname .. " > 0 in bias connection: %s\n",
		wname)
      end
      func(oldw, v, w, ann_component)
    end
  end
end

local function ann_optimizer_apply_constraints(opt,
					       wname, oldw, w,
					       ann_component)
  for hypname,func in pairs(opt.constraints) do
    local v = opt:get_option_of(wname, hypname)
    if v then
      -- sanity check
      if v > 0.0 and w:dim(2) == 1 then
	fprintf(io.stderr,
		"# WARNING!!! Possible " .. hypname .. " > 0 in bias connection: %s\n",
		wname)
      end
      func(w, oldw, v, ann_component)
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
  assert(self.valid_options[name], "Not recognized option " .. name)
  self.global_options[name] = value
  return self
end

function optimizer_methods:get_option(name,value)
  assert(self.valid_options[name], "Not recognized option " .. name)
  return self.global_options[name]
end

function optimizer_methods:set_layerwise_option(layer_name,name,value)
  assert(self.valid_options[name], "Not recognized option " .. name)
  self.layerwise_options[layer_name] = self.layerwise_options[layer_name] or {}
  self.layerwise_options[layer_name][name] = value
  return self
end

function optimizer_methods:get_layerwise_option(layer_name,name)
  assert(self.valid_options[name], "Not recognized option " .. name)
  return (self.layerwise_options[layer_name] or {})[name]
end

function optimizer_methods:get_option_of(layer_name,name)
  assert(self.valid_options[name], "Not recognized option " .. name)
  return ( (self.layerwise_options[layer_name] or {})[name] or
	     self.global_options[name] )
end

-- eval is a function which returns the data needed by the optimizer (at least,
-- the loss, the loss matrix for each pattern in the batch, the gradients, the
-- bunch size, and the ANN component)
--
-- cnn_table is a dictionary of connections objects, indexed by its names.
function optimizer_methods:execute(eval, cnn_table)
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

------------------------------------------------
------------------------------------------------
------------------------------------------------

local function ann_optimizer_apply_momentum(oldw, mt, w)
  if mt > 0.0 then
    -- intertia is computed between current and old weight matrices, but with
    -- inverted sign
    oldw:axpy(-1.0, w)
    -- apply momentum term to inertia computation, with inverted sign
    oldw:scal(-mt)
  else
    -- sets to ZERO the old matrix
    oldw:zeros()
  end
end

------------------------------------------------
--------- STOCHASTIC GRADIENT DESCENT ----------
------------------------------------------------

local sgd_methods, sgd_class_metatable = class("ann.optimizer.sgd",
					       ann.optimizer)

function sgd_class_metatable:__call(g_options, l_options, count)
  -- the base optimizer, with the supported learning parameters
  local obj = ann.optimizer({
			      {"learning_rate", "Learning speed factor (0.1)"},
			      {"momentum", "Learning inertia factor (0.1)"},
			      {"weight_decay", "Weight L2 regularization (1e-04)"},
			    },
			    g_options,
			    l_options,
			    count)
  obj = class_instance(obj, self)
  -- standard regularization and constraints
  obj:add_regularization("L1_norm", nil, "Weight L1 regularization (1e-05)")
  obj:add_constraint("max_norm_penalty", nil, "Weight max norm upper bound (4)")
  return obj
end

function sgd_methods:execute(eval, cnn_table)
  local arg = table.pack( eval() )
  local tr_loss,gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
  local bunch_size = bunch_size or 1
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  for cname,cnn in pairs(cnn_table) do
    local w,oldw     = cnn:matrix()
    local grad       = gradients[cname]
    local N          = cnn:get_shared_count()
    local lr         = assert(self:get_option_of(cname, "learning_rate"),
			      "The learning_rate parameter needs to be set")
    local mt         = self:get_option_of(cname, "momentum")     or 0.0
    local wd         = self:get_option_of(cname, "weight_decay") or 0.0
    local cwd        = 1.0 - wd
    --
    if wd > 0.0 and w:dim(2) == 1 then
      fprintf(io.stderr,
	      "# WARNING!!! Possible weight_decay > 0 in bias connection: %s\n",
	      cname)
    end
    --
    ann_optimizer_apply_momentum(oldw, mt, w)
    -- the weight decay SUMS the weight value. Other regularization is better to
    -- be after the back-propagation learning rule
    ann_optimizer_regularizations_weight_decay(oldw, cwd, w)
    -- apply back-propagation learning rule
    local norm_lr_rate = -1.0/math.sqrt( N * bunch_size ) * lr
    oldw:axpy(norm_lr_rate, grad)
    -- regularizations
    ann_optimizer_apply_regularizations(self, cname, oldw, w, ann_component)
    -- constraints
    ann_optimizer_apply_constraints(self, cname, oldw, w, ann_component)
    --
    -- swap current and old weight matrices
    cnn:swap()
    --
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      cnn:prune_subnormal_and_check_normal()
    end
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
  return obj
end

function sgd_methods:to_lua_string()
  local str_t = { "ann.optimizer.sgd(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ")" }
  return table.concat(str_t, "")
end

-----------------------------------
--------- RESILIENT PROP ----------
-----------------------------------

local rprop_methods, rprop_class_metatable = class("ann.optimizer.rprop",
						   ann.optimizer)

function rprop_class_metatable:__call(g_options, l_options, count,
				      steps, old_sign)
  -- the base optimizer, with the supported learning parameters
  local obj = ann.optimizer({
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
  obj.steps    = steps or {}
  obj.old_sign = old_sign or {}
  obj = class_instance(obj, self)
  obj:set_option("initial_step",  0.1)
  obj:set_option("eta_plus",      1.2)
  obj:set_option("eta_minus",     0.5)
  obj:set_option("max_step",      50)
  obj:set_option("min_step",      1e-05)
  obj:set_option("niter",         1)
  return obj
end

function rprop_methods:execute(eval, cnn_table)
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
    arg = table.pack( eval() )
    local tr_loss,gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
    -- the gradient computation could fail returning nil, it is important to
    -- take this into account
    if not gradients then return nil end
    --
    for cname,cnn in pairs(cnn_table) do
      local w,oldw     = cnn:matrix()
      steps[cname]     = steps[cname] or w:clone():fill(initial_step)
      local sign       = gradients[cname]:clone():sign()
      -- copy the weight
      oldw:copy(w)
      -- apply reprop learning rule
      if old_sign[cname] then
	ann_optimizer_utils.rprop.step(steps[cname],
				       old_sign[cname],
				       sign,
				       eta_minus,
				       eta_plus)
      end
      oldw:axpy(-1.0, sign:cmul(steps[cname]))
      -- keep the sign for the next iteration
      old_sign[cname] = sign
      --
      -- swap current and old weight matrices
      cnn:swap()
      --
      if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
	cnn:prune_subnormal_and_check_normal()
	collectgarbage("collect")
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
    obj.steps = table.map(self.steps, function(m) return m:clone() end)
  end
  if self.old_sign then
    obj.old_sign = table.map(self.old_sign, function(m) return m:clone() end)
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
		  "{",
		  iterator(pairs(self.steps)):
		  map(function(name,m)
			return string.format("[%q]",name),m:to_lua_string(format)
		      end):
		  concat("=",","),
		  "}",
		  ",",
		  "{",
		  iterator(pairs(self.old_sign)):
		  map(function(name,m)
			return string.format("[%q]",name),m:to_lua_string(format)
		      end):
		  concat("=",","),
		  "}",
		  ")" }
  return table.concat(str_t, "")
end

---------------------------------------
--------- CONJUGATE GRADIENT ----------
---------------------------------------

-- Conjugate Gradient implementation, copied/modified from optim package of
-- Torch 7, which is a rewrite of minimize.m written by Carl E. Rasmussen.
local cg_methods, cg_class_metatable = class("ann.optimizer.cg", ann.optimizer)

function cg_class_metatable:__call(g_options, l_options, count,
				   df0, df1, df2, df3, x0, s)
  -- the base optimizer, with the supported learning parameters
  local obj = ann.optimizer({
			      --			      {"momentum", "Learning inertia factor (0.1)"},
			      {"weight_decay", "Weights L2 regularization (1e-04)"},
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
  obj.state    = {
    df0 = df0,
    df1 = df1,
    df2 = df2, 
    df3 = df3,
    x0  = x0,
    s   = s,
  }
  obj = class_instance(obj, self)
  obj:set_option("rho",           0.01) -- rho is a constant in Wolfe-Powell conditions
  obj:set_option("sig",           0.5)  -- sig is another constant of Wolf-Powell
  obj:set_option("int",           0.1)  -- reevaluation limit
  obj:set_option("ext",           3.0)  -- maximum number of extrapolations
  obj:set_option("max_iter",      20)
  obj:set_option("ratio",         100)  -- maximum slope ratio
  -- standard regularization and constraints
  obj:add_regularization("L1_norm", nil, "Weight L1 regularization (1e-05)")
  obj:add_constraint("max_norm_penalty", nil, "Weight max norm upper bound (4)")
  return obj
end

function cg_methods:execute(eval, cnn_table)
  -- COPY function
  local copy = function(dest,source)
    iterator(pairs(dest)):apply(function(k,v) v:copy(source[k]) end)
    return dest
  end
  -- DOT_REDUCE function
  local dot_reduce = function(t1,t2)
    local t2 = t2 or t1
    if t1 ~= t2 then
      return iterator(pairs(t1)):
      map(function(k,v) return v:contiguous(),t2[k]:contiguous() end):
      map(function(v1,v2) return v1:rewrap(v1:size()),v2:rewrap(v2:size()) end):
      map(function(v1,v2) return v1:dot(v2) end):
      reduce(math.add(), 0)
    else
      return iterator(pairs(t1)):
      map(function(k,v) return v:contiguous():rewrap(v:size()) end):
      map(function(v) return v:dot(v) end):
      reduce(math.add(), 0)
    end
  end
  -- ADD function
  local add = function(x, z, s)
    iterator(pairs(x)):apply(function(k,v) v:axpy(z, s[k]) end)
    return x
  end
  -- SCAL function
  local scal = function(t, ss)
    iterator(pairs(t)):apply(function(k,v) v:scal(ss) end)
    return t
  end
  -- CLONE function
  local clone = function(t)
    return iterator(pairs(t)):map(function(k,v)
				    return k,v:clone()
				  end):table()
  end
  -- CLONE_ONLY_DIMS function
  local clone_only_dims = function(t)
    return iterator(pairs(t)):
    map(function(k,v) return k,matrix.col_major(table.unpack(v:dim())) end):
    table()
  end
  -- UPDATE_GRADIENTS function
  local update_gradients = function(gradients)
    return gradients
    -- for cname,cnn in pairs(cnn_table) do
    --   local w,oldw        = cnn:matrix()
    --   local grad          = gradients[cname]
    --   local mt  = self:get_option_of(cname, "momentum") or 0.0
    --   --
    --   if mt > 0.0 then
    -- 	local aux = w:clone():axpy(-1.0, oldw)
    -- 	grad:axpy(mt, aux)
    --   end
    --   --
    -- end
  end
  -- COPY_WEIGHTS function
  local copy_weights = function(t)
    iterator(pairs(cnn_table)):map(function(k,v)
				     local w,oldw=v:matrix()
				     t[k]:copy(w)
				     return k,w
				   end):table()
  end
  -- UPDATE_WEIGHTS function
  local update_weights = function(x, dir, s)
    -- count one more update iteration
    self:count_one()
    for cname,cnn in pairs(cnn_table) do
      local w,oldw        = cnn:matrix()
      local grad          = s[cname]
      local wd  = self:get_option_of(cname, "weight_decay") or 0.0
      local cwd = 1.0 - wd
      --
      if wd > 0.0 and w:dim(2) == 1 then
	fprintf(io.stderr,
		"# WARNING!!! Possible weight_decay > 0 in bias connection: %s\n",
		cname)
      end
      --
      oldw:zeros()
      -- the weight decay SUMS the weight value. Other regularization is better to
      -- be after the back-propagation learning rule
      ann_optimizer_regularizations_weight_decay(oldw, cwd, w)
      -- apply back-propagation learning rule
      oldw:axpy(dir, grad)
      -- regularizations
      ann_optimizer_apply_regularizations(self, cname, oldw, w, ann_component)
      -- constraints
      ann_optimizer_apply_constraints(self, cname, oldw, w, ann_component)
      --
      -- swap current and old weight matrices
      cnn:swap()
      --
      if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
	iterator(pairs(cnn_table)):select(2):call(prune_subnormal_and_check_normal)
      end
    end
    -- swap in x
    x.w,x.oldw = x.oldw,x.w
  end
  -- COPY_WEIGHTS function
  local copy_weights = function(t)
    iterator(pairs(cnn_table)):map(function(k,v)
				     local w,oldw=v:matrix()
				     t[k]:copy(w)
				     return k,w
				   end):table()
  end
  ----------------------------------------------------------------------------
  
  local x = {
    
    w = iterator(pairs(cnn_table)):
    map(function(k,v) return k,v:matrix() end):
    select(1,2):table(),
    
    oldw = iterator(pairs(cnn_table)):
    map(function(k,v) return k,v:matrix() end):
    select(1,3):table()
    
  }
  local rho           = self:get_option("rho")
  local sig           = self:get_option("sig")
  local int           = self:get_option("int")
  local ext           = self:get_option("ext")
  local max_iter      = self:get_option("max_iter")
  local ratio         = self:get_option("ratio")
  local max_eval      = self:get_option("max_eval") or max_iter*1.25
  local red           = 1
  
  local i             = 0
  local ls_failed     = 0
  local fx            = {}

  -- we need three points for the interpolation/extrapolation stuff
  local z1,z2,z3 = 0,0,0
  local d1,d2,d3 = 0,0,0
  local f1,f2,f3 = 0,0,0

  local df1 = self.state.df1 or clone_only_dims(x.w)
  local df2 = self.state.df2 or clone_only_dims(x.w)
  local df3 = self.state.df3 or clone_only_dims(x.w)
  
  -- search direction
  local s = self.state.s or clone_only_dims(x.w)
  
  -- we need a temp storage for X
  local x0  = self.state.x0 or { w=clone(x.w), oldw=clone(x.oldw) }
  local f0  = 0
  local df0 = self.state.df0 or clone_only_dims(x.w)
  
  -- evaluate at initial point
  local arg = table.pack( eval() )
  local tr_loss,gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
  update_gradients(gradients)
  f1 = tr_loss
  table.insert(fx, f1)
  copy(df1,gradients)
  i=i+1
  
  -- initial search direction
  copy(s,df1)
  scal(s,-1)
  
  -- slope
  d1 = -dot_reduce(s)
  -- initial step
  z1 = red/(1-d1)
  
  while i < math.abs(max_eval) do
    
    copy(x0.w,    x.w)
    copy(x0.oldw, x.oldw)
    
    f0 = f1
    copy(df0,df1)
    
    update_weights(x, z1, s)

    arg = table.pack( eval() )
    tr_loss,gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
    update_gradients(gradients)
    f2 = tr_loss
    
    copy(df2,gradients)
    i=i+1
    d2 = dot_reduce(df2,s)
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
	arg = table.pack( eval() )
	tr_loss,gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
	update_gradients(gradients)
	f2 = tr_loss
	copy(df2,gradients)
	i=i+1
	m = m - 1
	d2 = dot_reduce(df2,s)
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
      
      arg = table.pack( eval() )
      tr_loss,gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
      update_gradients(gradients)
      f2 = tr_loss
      copy(df2, gradients)
      i=i+1
      m = m - 1
      d2 = dot_reduce(df2,s)
    end
    if success then
      f1 = f2
      table.insert(fx, f1)
      local ss = (dot_reduce(df2,df2) - dot_reduce(df2,df1))/dot_reduce(df1,df1)
      scal(s,ss)
      add(s,-1,df2)
      df1,df2 = df2,df1
      -- local tmp = clone(df1)
      -- copy(df1,df2)
      -- copy(df2,tmp)
      d2 = dot_reduce(df1,s)
      if d2> 0 then
	copy(s,df1)
	scal(s,-1)
	d2 = -dot_reduce(s,s)
      end
      z1 = z1 * math.min(ratio, d1/(d2-1e-320))
      d1 = d2
      ls_failed = 0
    else
      copy(x.w,    x0.w)
      copy(x.oldw, x0.oldw)
      f1 = f0
      copy(df1,df0)
      if ls_failed or i>max_eval then
	break
      end
      df1,df2 = df2,df1
      -- local tmp = clone(df1)
      -- copy(df1,df2)
      -- copy(df2,tmp)
      copy(s,df1)
      scal(s,-1)
      d1 = -dot_reduce(s,s)
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
  
  -- evaluate the function at the end
  local arg = table.pack( eval() )
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
    obj.state.df0 = table.map(self.state.df0, function(m) return m:clone() end)
  end
  if self.state.df1 then
    obj.state.df1 = table.map(self.state.df1, function(m) return m:clone() end)
  end
  if self.state.df2 then
    obj.state.df2 = table.map(self.state.df2, function(m) return m:clone() end)
  end
  if self.state.df3 then
    obj.state.df3 = table.map(self.state.df3, function(m) return m:clone() end)
  end
  if self.state.x0 then
    obj.state.x0 = {
      w = table.map(self.state.x0.w, function(m) return m:clone() end),
      oldw = table.map(self.state.x0.oldw, function(m) return m:clone() end),
    }
  end
  if self.state.s then
    obj.state.s = table.map(self.state.s, function(m) return m:clone() end)
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
