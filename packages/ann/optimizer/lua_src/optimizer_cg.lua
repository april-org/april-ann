local assert = assert
local ipairs = ipairs
local math = math
local pairs = pairs
local string = string
local table = table
local type = type
--
local april_assert = april_assert
local get_table_fields = get_table_fields
local iterator = iterator
local mop = matrix.op
local md = matrix.dict

local MAX_UPDATES_WITHOUT_PRUNE = ann.optimizer.MAX_UPDATES_WITHOUT_PRUNE

---------------------------------------
--------- CONJUGATE GRADIENT ----------
---------------------------------------

-- Conjugate Gradient implementation, rewrite from optim package of
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
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"L1_norm", "Weight L1 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
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
  if not g_options then
    self:set_option("rho",           0.01) -- rho is a constant in Wolfe-Powell conditions
    self:set_option("sig",           0.5)  -- sig is another constant of Wolf-Powell
    self:set_option("int",           0.1)  -- reevaluation limit
    self:set_option("ext",           3.0)  -- maximum number of extrapolations
    self:set_option("max_iter",      20)
    self:set_option("ratio",         100)  -- maximum slope ratio
    self:set_option("weight_decay",  0.0)
    self:set_option("L1_norm",       0.0)
    self:set_option("max_norm_penalty", 0.0)
  end
end

function cg_methods:execute(eval, weights)
  local table = table
  local assert = assert
  local math = math
  --
  local origw = weights
  -- DO EVAL
  local do_eval = function(x,i)
    local arg = table.pack( eval(x, i) )
    local tr_loss,gradients = table.unpack(arg)
    local reg = 0.0
    for wname,w in pairs(x) do
      local l1 = self:get_option_of(wname, "L1_norm")
      local l2 = self:get_option_of(wname, "weight_decay")
      if l1 > 0.0 then reg = reg + l1*mop.abs(w):sum() end
      if l2 > 0.0 then reg = reg + 0.5*l2*w:dot(w) gradients[wname]:axpy(l2, w) end
    end
    arg[1] = arg[1] + reg
    return arg
  end
  -- UPDATE_WEIGHTS function
  local update_weights = function(x, dir, s)
    md.axpy(x, dir, s)
    for wname,w in md.iterator(x) do
      local l1 = self:get_option_of(wname, "L1_norm")
      -- L1 regularization, truncated gradient implementation
      if l1 > 0.0 then ann.optimizer.utils.l1_truncate_gradient(w, math.abs(dir)*l1) end
    end
  end
  -- APPLY REGULARIZATION AND PENALTIES
  local apply_penalties = function(x)
    for wname,w in md.iterator(x) do
      local mnp = self:get_option_of(wname, "max_norm_penalty")
      -- constraints
      if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
    end
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      md.prune_subnormal_and_check_normal( x )
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

  local df1 = self.state.df1 or md.clone_only_dims( x )
  local df2 = self.state.df2 or md.clone_only_dims( x )
  local df3 = self.state.df3 or md.clone_only_dims( x )
  
  -- search direction
  local s = self.state.s or md.clone_only_dims( x )
  
  -- we need a temp storage for X
  local x0  = self.state.x0 or md.clone( x )
  local f0  = 0
  local df0 = self.state.df0 or md.clone_only_dims( x )
  
  -- evaluate at initial point
  local arg = do_eval(origw, i)
  local tr_loss,gradients = table.unpack(arg)
  if not gradients then return nil end
  f1 = tr_loss
  table.insert(fx, f1)
  md.copy(df1, gradients)
  i=i+1
  
  -- initial search direction
  md.scal( md.copy(s, df1), -1 )
  
  -- slope
  d1 = -md.dot(s, s)
  -- initial step
  z1 = red/(1-d1)
  
  while i < math.abs(max_eval) do
    
    md.copy( x0, x )
    
    f0 = f1
    md.copy( df0, df1 )
    
    update_weights(x, z1, s)

    arg = do_eval(origw, i)
    tr_loss,gradients = table.unpack(arg)
    f2 = tr_loss
    
    md.copy( df2, gradients )
    i=i+1
    d2 = md.dot( df2, s )
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
        arg = do_eval(origw, i)
	tr_loss,gradients = table.unpack(arg)
	f2 = tr_loss
	md.copy( df2, gradients )
	i=i+1
	m = m - 1
	d2 = md.dot( df2, s )
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
      
      arg = do_eval(origw, i)
      tr_loss,gradients = table.unpack(arg)
      f2 = tr_loss
      md.copy( df2, gradients )
      i=i+1
      m = m - 1
      d2 = md.dot( df2, s )
    end
    if success then
      f1 = f2
      table.insert(fx, f1)
      local ss = ( md.dot( df2, df2 ) - md.dot( df2, df1 ))/md.dot( df1, df1 )
      md.scal( s, ss )
      md.axpy( s, -1, df2 )
      df1,df2 = df2,df1
      -- local tmp = clone(df1)
      -- copy(df1,df2)
      -- copy(df2,tmp)
      d2 = md.dot( df1, s )
      if d2 > 0 then
	md.copy( s, df1 )
	md.scal( s, -1 )
	d2 = -md.dot( s, s )
      end
      z1 = z1 * math.min(ratio, d1/(d2 - FLT_MIN))
      d1 = d2
      ls_failed = 0
    else
      md.copy( x, x0 )
      f1 = f0
      md.copy( df1, df0 )
      if ls_failed or i>max_eval then
	break
      end
      df1,df2 = df2,df1
      -- local tmp = clone(df1)
      -- copy(df1,df2)
      -- copy(df2,tmp)
      md.copy( s, df1 )
      md.scal( s, -1 )
      d1 = -md.dot( s, s )
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
  
  apply_penalties(x)
  
  -- evaluate the function at the end
  local arg = do_eval(origw, i)
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
    obj.state.df0 = md.clone( self.state.df0 )
  end
  if self.state.df1 then
    obj.state.df1 = md.clone( self.state.df1 )
  end
  if self.state.df2 then
    obj.state.df2 = md.clone( self.state.df2 )
  end
  if self.state.df3 then
    obj.state.df3 = md.clone( self.state.df3 )
  end
  if self.state.x0 then
    obj.state.x0 = md.clone( self.state.x0 )
  end
  if self.state.s then
    obj.state.s = md.clone( self.state.s )
  end
  return obj
end

function cg_methods:ctor_name()
  return "ann.optimizer.cg"
end
function cg_methods:ctor_params()
  return self.global_options,
  self.layerwise_options,
  self.count,
  self.aw
end

local cg_properties = {
  gradient = true
}
function cg_methods:needs_property(name)
  return cg_properties[name]
end
