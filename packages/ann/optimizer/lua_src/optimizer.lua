local math = math
local table = table
local string = string
--
local ipairs = ipairs
local pairs = pairs
local assert = assert
--
local type = type
local mop = matrix.op
local iterator = iterator
local get_table_fields = get_table_fields
local april_assert = april_assert
local wrap_matrices = matrix.dict.wrap_matrices

local FLT_MIN = mathcore.limits.float.min()

ann.optimizer = ann.optimizer or {}
ann.optimizer.MAX_UPDATES_WITHOUT_PRUNE = 100

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
local ann_optimizer_utils = ann.optimizer.utils

-- this function receives a weights matrix, a L1 regularization parameter (can
-- be a number of a matrix with L1 for every weight), and an optional update
-- matrix (for momentum purposes, it can be nil)
function ann_optimizer_utils.l1_truncate_gradient(w, l1, update)
  local z = mop.abs(w):gt(l1):to_float() -- which weights won't cross zero
  -- compute L1 update
  local u = mop.sign(w)
  if type(l1) == "number" then u:scal(l1) else u:cmul(l1) end
  -- apply L1 update to weights
  w:axpy(-1.0, u)
  if update then
    -- apply L1 update to update matrix (for momentum)
    update:axpy(-1.0, u)
  end
  -- remove weights which cross zero
  w:cmul(z)
end

-- receives a weights matrix and a max norm penalty value
function ann_optimizer_utils.max_norm_penalty(w, mnp)
  for _,row in matrix.ext.iterate(w,1) do
    local n2 = row:norm2()
    if n2 > mnp then row:scal(mnp / n2) end
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

function optimizer_methods:has_property(name)
  return false
end

------------------------------------------------
------------------------------------------------
------------------------------------------------




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
  self:add_constraint("L1_norm", nil, "Weight L1 regularization (1e-05)")
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
  s:copy(df1):scal(-1)
  
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
      z1=z1+z2
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
      if d2 > 0 then
	s:copy(df1)
	s:scal(-1)
	d2 = -s:dot(s)
      end
      z1 = z1 * math.min(ratio, d1/(d2 - FLT_MIN))
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
