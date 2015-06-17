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
  local z = mop.abs(w):gt(l1):convert_to("float") -- which weights won't cross zero
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
