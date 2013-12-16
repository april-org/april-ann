autodiff    = autodiff or {}
autodiff.op = autodiff.op or {}

local CONSTANT = 'constant'
local SCALAR   = 'scalar'

local SYMBOLS = {}

local function infer(...)
  local arg = table.pack(...)
  local dtype
  if type(arg[1]) == "number" then
    dtype = CONSTANT
  else
    dtype = arg[1].dtype
  end
  for i=2,#arg do
    local argi_dtype = SCALAR
    if type(arg[i]) == "number" then arg[i] = CONSTANT end
    if argi_dtype == SCALAR then dtype = SCALAR end
  end
  return dtype
end

local symbol_mt = {
  __call = function(s,...) return s:eval(...) end,
  __add  = function(a,b) return autodiff.op[ infer(a,b) ].add(a,b) end,
  __sub  = function(a,b) return autodiff.op[ infer(a,b) ].sub(a,b) end,
  __mul  = function(a,b) return autodiff.op[ infer(a,b) ].mul(a,b) end,
  __div  = function(a,b) return autodiff.op[ infer(a,b) ].div(a,b) end,
  __unm  = function(a)   return autodiff.op[ infer(a) ].unm(a)     end,
  __pow  = function(a,b) return autodiff.op[ infer(a,b) ].pow(a,b) end,
  __tostring = function(s) return s.name end,
}

local function symbol(name,dtype)
  local t
  if SYMBOLS[name] then t = SYMBOLS[name]
  else
    t = {
      name  = name,
      dtype = dtype,
      issymbol = true,
      eval = function(self,values)
	return assert(values[self.name], "Undefined value " .. self.name)
      end
    }
    SYMBOLS[name] = t
    setmetatable(t, symbol_mt)
  end
  return t
end

local function op(name, dtype, args, eval, diff)
  local s = symbol(string.format("(%s %s)", name,
				 iterator(ipairs(args)):select(2):
				 map(tostring):concat(" ")),
		   dtype)
  s.isop = true
  s.args = args
  s.eval = function(self, values, prev_cache)
    local cache = prev_cache or {}
    local v = values[self.name] or cache[self.name] or eval(self, values, cache)
    cache[self.name] = v
    return v
  end
  s.diff = function(self, target)
    return diff(self, (type(target)=="string" and target) or target.name)
  end
  return s
end

-----------------------------------------------------------------------------

function autodiff.clear()
  SYMBOLS = {}
end

function autodiff.symbol(names,dtype)
  local result = iterator(names:gmatch("[^%s]+")):
  map(function(name) return symbol(name,dtype) end):table()
  return table.unpack(result)
end

function autodiff.func(s,args,shared_values)
  local args,shared_values = args or {},shared_values or {}
  for i,s in ipairs(args) do
    assert(type(s)=="table" and s.issymbol,
	   "Argument " .. i .. " is not a symbol")
  end
  for name,_ in pairs(shared_values) do
    assert(SYMBOLS[name], "Undefined symbol " .. name)
  end
  return function(...)
    local args2 = table.pack(...)
    assert(#args == #args2,
	   string.format("Incorrect number of arguments, expected %d, found %d\n",
			 #args, #args2))
    local values = iterator(ipairs(args)):
    map(function(k,v) return v.name,args2[k] end):table()
    for k,v in pairs(shared_values) do values[k] = v end
    return s:eval(values)
  end
end

setmetatable(autodiff.op,
	     {
	       __index = function(s,key)
		 return rawget(s,key) or
		   function(...)
		     local dtype = infer(...)
		     local t = assert(autodiff.op[dtype],
				      "Incorrect type " .. dtype)
		     local t = assert(autodiff.op[dtype][key],
				      "Operation: " .. key .. " not implemented for type: " .. dtype)
		     return t(...)
		   end
	       end,
	     })

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- CONSTANTS

autodiff.constant = function(...)
  local arg = table.pack(...)
  local result = {}
  for _,value in ipairs(arg) do
    local s = autodiff.symbol(tostring(value), CONSTANT)
    s.value = value
    s.eval  = function(self) return self.value end
    s.diff  = function(self) return autodiff.constant( 0 ) end
    table.insert(result, s)
  end
  return table.unpack(result)
end

-- CONSTANT OPERATIONS

local function coercion(a)
  if type(a) == "number" then return autodiff.constant(a)
  else return a
  end
end

autodiff.op[CONSTANT] = {
  
  add = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a() + b() ) end,
  sub = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a() - b() ) end,
  pow = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a() ^ b() ) end,
  unm = function(a)   local a=coercion(a) return autodiff.constant( - a() )     end,
  mul = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a() * b() ) end,
  div = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a() / b() ) end,

  log = function(a) local a=coercion(a) return autodiff.constant( math.log( a() ) ) end,
  exp = function(a) local a=coercion(a) return autodiff.constant( math.exp( a() ) ) end,
  sin = function(a) local a=coercion(a) return autodiff.constant( math.sin( a() ) ) end,
  cos = function(a) local a=coercion(a) return autodiff.constant( math.cos( a() ) ) end,

}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- SCALARS

autodiff[SCALAR] = function(names)
  local t = table.pack(autodiff.symbol(names, SCALAR))
  for i=1,#t do
    t[i].diff = function(self, target)
      local tname = (type(target)=="string" and target) or target.name
      if tname == self.name then
	return autodiff.constant(1)
      else 
	return autodiff.constant(0)
      end
    end
  end
  return table.unpack(t)
end

-- CONSTANT OPERATIONS

autodiff.op[SCALAR] = {
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = op('add', SCALAR, {a,b},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   local b = self.args[2]:eval(...)
		   return a + b
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local da = self.args[1]:diff(target)
		   local db = self.args[2]:diff(target)
		   return da + db
		 end)
    return s
  end,
  
  sub = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a + (-1 * b)
  end,

  mul = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = op('mul', SCALAR, {a,b},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   local b = self.args[2]:eval(...)
		   return a * b
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local a,b = self.args[1],self.args[2]
		   local da,db = a:diff(target),b:diff(target)
		   return da*b + a*db
		 end)
    return s
  end,
  
  div = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a * (b^(-1))
  end,

  pow = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = op('pow', SCALAR, {a,b},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   local b = self.args[2]:eval(...)
		   return a^b
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local a,b = self.args[1],self.args[2]
		   local da  = a:diff(target)
		   return b * (a^(b-1)) * da
		 end)
    return s
  end,
  
  unm = function(a)
    local a = coercion(a)
    return (-1) * a
  end,

  log = function(a)
    local a = coercion(a)
    local s = op('log', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.log(a)
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local a  = self.args[1]
		   local da = a:diff(target)
		   return 1/a * da
		 end)
    return s
  end,

  exp = function(a)
    local a = coercion(a)
    local s = op('exp', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.exp(a)
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local a  = self.args[1]
		   local da = a:diff(target)
		   return autodiff.op.exp(a) * da
		 end)
    return s
  end,

  cos = function(a)
    local a = coercion(a)
    local s = op('cos', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.cos(a)
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local a  = self.args[1]
		   local da = a:diff(target)
		   return autodiff.op.sin(a) * da
		 end)
    return s
  end,

  sin = function(a)
    local a = coercion(a)
    local s = op('sin', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.sin(a)
		 end,
		 function(self, target)
		   if self.name == target then return autodiff.constant(1) end
		   local a  = self.args[1]
		   local da = a:diff(target)
		   return autodiff.op.cos(a) * da
		 end)
    return s
  end,

}
