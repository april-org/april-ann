local MAX_UPDATES_WITHOUT_PRUNE=100

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function ann_optimizer_regularizations_weight_decay(oldw, cwd, w)
  -- sum current weights by applying the complementary of weight decay term
  oldw:axpy(cwd, w)  
end

------------------------------------------------------------------------------

get_table_from_dotted_string("ann.optimizer.regularizations", true)

function ann.optimizer.regularizations.L1_norm(oldw, value, w)
  if value > 0.0 then
    -- sum current weights by applying the complementary of weight decay term
    oldw:map(function(x)
	       print(x)
	       if x > 0 then return x-value
	       elseif x < 0 then return x+value
	       else return 0
	       end
	     end)
  end
end

------------------------------------------------------------------------------

get_table_from_dotted_string("ann.optimizer.constraints", true)

function ann.optimizer.constraints.max_norm_penalty(oldw, w, mp)
  if mp > 0.0 then
    local old_sw     = oldw:sliding_window()
    local old_window = nil
    local sw         = w:sliding_window()
    local window     = nil
    while not sw:is_end() do
      old_window  = old_sw:get_matrix(old_window)
      local norm2 = old_window:norm2()
      if norm2 > mp then
	local scal_factor = mp / norm2
	window = sw:get_matrix(window)
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
			       valid_options     = iterator(ipairs(valid_options or {})):map(function(i,name)return name,true end):table(),
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
-- func(oldw, value, w, ann_component) where oldw is the destination matrix
function optimizer_methods:add_regularization(hyperparameter_name, func)
  assert(not self.valid_options[hyperparameter_name],
	 "Redefinition of hyperparameter " .. hyperparameter_name)
  self.valid_options[hyperparameter_name] = true
  self.regularizations[hyperparameter_name] = func or ann.optimizer.regularizations[hyperparameter_name]
end

-- constraint functions has the API:
-- func(oldw, value, w, ann_component) => remember to apply the same constraint to oldw and w
function optimizer_methods:add_constraint(hyperparameter_name, func)
  assert(not self.valid_options[hyperparameter_name],
	 "Redefinition of hyperparameter " .. hyperparameter_name)
  self.valid_options[hyperparameter_name] = true
  self.constraints[hyperparameter_name] = func or ann.optimizer.constraints[hyperparameter_name]
end

local function ann_optimizer_apply_regularizations(opt,
						   wname, oldw, w,
						   ann_component)
  for hypname,func in ipairs(opt.regularizations) do
    local v = opt:get_option_of(wname, hypname)
    if v then
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
  for hypname,func in ipairs(opt.constraints) do
    local v = opt:get_option_of(wname, hypname)
    if v then
      if v > 0.0 and w:dim(2) == 1 then
	fprintf(io.stderr,
		"# WARNING!!! Possible " .. hypname .. " > 0 in bias connection: %s\n",
		wname)
      end
      func(oldw, v, w, ann_component)
    end
  end
end

function optimizer_methods:show_options()
  local t = iterator(pairs(self.valid_options)):select(1):enumerate():table()
  table.sort(t)
  iterator(ipairs(t)):apply(print)
end

function optimizer_methods:set_option(name,value)
  assert(self.valid_options[name], "Not recognized option " .. name)
  self.global_options[name] = value
end

function optimizer_methods:get_option(name,value)
  assert(self.valid_options[name], "Not recognized option " .. name)
  return self.global_options[name]
end

function optimizer_methods:set_layerwise_option(layer_name,name,value)
  assert(self.valid_options[name], "Not recognized option " .. name)
  self.layerwise_options[layer_name] = self.layerwise_options[layer_name] or {}
  self.layerwise_options[layer_name][name] = value
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
-- the gradients, the bunch size, the loss matrix, and the ANN component)
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
			      "learning_rate",
			      "momentum",
			      "weight_decay",
			    },
			    g_options,
			    l_options,
			    count)
  obj = class_instance(obj, self)
  -- standard regularization and constraints
  obj:add_constraint("max_norm_penalty")
  obj:add_regularization("L1_norm")
  return obj
end

function sgd_methods:execute(eval, cnn_table)
  local arg = table.pack( eval() )
  local gradients,bunch_size,tr_loss_matrix,ann_component = table.unpack(arg)
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
    ann_optimizer_regularizations_weight_decay(oldw, cwd, w)
    --
    ann_optimizer_apply_regularizations(self, cname, oldw, w, ann_component)
    -- apply back-propagation learning rule
    local norm_lr_rate = -1.0/math.sqrt( N * bunch_size ) * lr
    oldw:axpy(norm_lr_rate, grad)
    --
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

-- ---------------------------------------
-- --------- CONJUGATE GRADIENT ----------
-- ---------------------------------------

-- -- This implementation is based in the code of optim package of Torch 7:
-- -- http://github.com/torch/optim/blob/master/cg.lua

-- local cg_methods, cg_class_metatable = class("ann.optimizer.cg",
-- 					     ann.optimizer)

-- function cg_class_metatable:__call(g_options, l_options, count)
--   -- the base optimizer, with the supported learning parameters
--   local obj = ann.optimizer({
-- 			      "max_eval",
-- 			      "max_iter",
-- 			      "rho",
-- 			      "sig",
-- 			      "int",
-- 			      "ext",
-- 			      "ratio",
-- 			      "learning_rate",
-- 			      "momentum",
-- 			      "weight_decay",
-- 			      "max_norm_penalty"
-- 			    },
-- 			    g_options,
-- 			    l_options,
-- 			    count)
--   obj = class_instance(obj, self)
--   return obj
-- end

-- local function clone_matrix(t)
--   return iterator(pairs(t)):map(function(name,c)return name,(c:matrix()):clone()end):table()
-- end

-- local function clone(t)
--   return iterator(pairs(t)):map(function(name,c)return name,c:clone()end):table()
-- end

-- local function copy(orig,dest)
--   apply(function(name,origc,destc)
-- 	  local orig_w,orig_oldw = origc:matrix()
-- 	  local dest_w,dest_oldw = destc:matrix()
-- 	  dest_w:copy(orig_w)
-- 	  dest_oldw:copy(orig_oldw)
-- 	end)
-- end

-- local function table_apply(t, func)
--   iterator(pairs(t)):select(2):apply(func)
-- end

-- function cg_methods:execute(eval, cnn_table)
--   local rho = self:get_option("rho") or 0.01
--   local sig = self:get_option("sig") or 0.5
--   local int = self:get_option("int") or 0.1
--   local ext = self:get_option("ext") or 3.0
--   local max_iter = self:get_option("max_iter") or 20
--   local max_eval = self:get_option("max_eval") or max_iter*1.25
--   local ratio = self:get_option("ratio") or 100
--   local verbose = self:get_option("verbose")
--   local red = 1
  
--   local i = 0
--   local ls_failed = false
--   local fx = {}
  
--   -- three points for the interpolation/extrapolation
--   local z1,z2,z3=0,0,0
--   local d1,d2,d3=0,0,0
--   local f1,f2,f3=0,0,0
  
--   local df1 = self.df1 or clone_matrix(cnn_table)
--   local df2 = self.df2 or clone_matrix(cnn_table)
--   local df3 = self.df3 or clone_matrix(cnn_table)
--   local tdf
  
--   -- search direction
--   local s = clone_matrix(cnn_table)
  
--   -- temporal storage for the connections
--   local x0 = clone(cnn_table)
--   local f0 = 0
--   local df0 = self.df0 or clone_matrix(cnn_table)
  
--   -- evaluate at initial point
--   local arg = table.pack( eval() )
--   local gradients,bunch_size,tr_loss_matrix = table.unpack(arg)

--   table.insert(fx, tr_loss_matrix:sum()/bunch_size)
--   copy(df1, gradients)
--   i=i+1
  
--   -- initial search direction
--   copy(df1, s)
--   table_apply(s, function(name,m) m:scal(-1) end)
  
--   d1 = -s:dot
  

--   for cname,cnn in pairs(cnn_table) do
--     local w,oldw     = cnn:matrix()
--     local grad       = gradients[cname]
--     local N          = cnn:get_shared_count()
--     local lr         = assert(self:get_option_of(cname, "learning_rate"),
-- 			      "The learning_rate parameter needs to be set")
--     local mt         = self:get_option_of(cname, "momentum")     or 0.0
--     local wd         = self:get_option_of(cname, "weight_decay") or 0.0
--     local cwd        = 1.0 - wd
--     local mp         = self:get_option_of(cname, "max_norm_penalty") or -1.0
--     --
--     ann.optimizer.apply_momentum(oldw, mt, w)
--     ann.optimizer.apply_weight_decay(oldw, cwd, w)
--     --
--     -- apply back-propagation learning rule
--     local norm_lr_rate = -1.0/math.sqrt( N * bunch_size ) * lr
--     oldw:axpy(norm_lr_rate, grad)
--     --
--     ann.optimizer.apply_max_norm_penalty(oldw, w, mp)
--     --
--     -- swap current and old weight matrices
--     cnn:swap()
--     --
--     if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
--       cnn:prune_subnormal_and_check_normal()
--     end
--   end
--   -- count one more update iteration
--   self:count_one()
--   -- returns the same as returned by eval()
--   return table.unpack(arg)
-- end

-- function cg_methods:clone()
--   local obj = ann.optimizer.cg()
--   obj.count             = self.count
--   obj.layerwise_options = table.deep_copy(self.layerwise_options)
--   obj.global_options    = table.deep_copy(self.global_options)
--   return obj
-- end

-- function cg_methods:to_lua_string()
--   local str_t = { "ann.optimizer.cg(",
-- 		  table.tostring(self.global_options),
-- 		  ",",
-- 		  table.tostring(self.layerwise_options),
-- 		  ",",
-- 		  tostring(self.count),
-- 		  ")" }
--   return table.concat(str_t, "")
-- end
