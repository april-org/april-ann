AD    = AD or {}
AD.op = AD.op or {}

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
  __add  = function(a,b) return AD.op[ infer(a,b) ].add(a,b) end,
  __sub  = function(a,b) return AD.op[ infer(a,b) ].sub(a,b) end,
  __mul  = function(a,b) return AD.op[ infer(a,b) ].mul(a,b) end,
  __div  = function(a,b) return AD.op[ infer(a,b) ].div(a,b) end,
  __unm  = function(a)   return AD.op[ infer(a) ].unm(a)     end,
  __pow  = function(a,b) return AD.op[ infer(a,b) ].pow(a,b) end,
  __tostring = function(s) return s.name end,
}

local function symbol(name,dtype)
  local t
  if SYMBOLS[name] then t = SYMBOLS[name]
  else
    t = {
      name  = name,
      dtype = dtype,
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
  return s
end

-----------------------------------------------------------------------------

function AD.clear()
  SYMBOLS = { }
end

function AD.symbol(names,dtype)
  local result = iterator(names:gmatch("[^%s]+")):
  map(function(name) return symbol(name,dtype) end):table()
  return table.unpack(result)
end

function AD.func(s,...)
  local arg = table.pack(...)
  return function(...)
    local arg2 = table.pack(...)
    assert(#arg == #arg2,
	   string.format("Incorrect number of arguments, expected %d, found %d\n",
			 #arg, #arg2))
    return s:eval( iterator(ipairs(arg)):
		   map(function(k,v) return v.name,arg2[k] end):
		   table() )
  end
end

setmetatable(AD.op,
	     {
	       __index = function(s,key)
		 return rawget(s,key) or
		   function(...)
		     local dtype = infer(...)
		     local t = assert(AD.op[dtype],
				      "Incorrect type " .. dtype)
		     local t = assert(AD.op[dtype][key],
				      "Operation: " .. key .. " not implemented for type: " .. dtype)
		     return t(...)
		   end
	       end,
	     })

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- CONSTANTS

AD.constant = function(...)
  local arg = table.pack(...)
  local result = {}
  for _,value in ipairs(arg) do
    local s = AD.symbol(tostring(value), CONSTANT)
    s.value = value
    s.eval  = function(self) return self.value end
    s.diff  = function(self) return AD.constant( 0 ) end
    table.insert(result, s)
  end
  return table.unpack(result)
end

-- CONSTANT OPERATIONS

local function coercion(a)
  if type(a) == "number" then return AD.constant(a)
  else return a
  end
end

AD.op[CONSTANT] = {
  
  add = function(a,b) local a,b=coercion(a),coercion(b) return AD.constant( a() + b() ) end,
  sub = function(a,b) local a,b=coercion(a),coercion(b) return AD.constant( a() - b() ) end,
  pow = function(a,b) local a,b=coercion(a),coercion(b) return AD.constant( a() ^ b() ) end,
  unm = function(a)   local a=coercion(a) return AD.constant( - a() )     end,
  mul = function(a,b) local a,b=coercion(a),coercion(b) return AD.constant( a() * b() ) end,
  div = function(a,b) local a,b=coercion(a),coercion(b) return AD.constant( a() / b() ) end,

  log = function(a) local a=coercion(a) return AD.constant( math.log( a() ) ) end,
  exp = function(a) local a=coercion(a) return AD.constant( math.exp( a() ) ) end,
  sin = function(a) local a=coercion(a) return AD.constant( math.sin( a() ) ) end,
  cos = function(a) local a=coercion(a) return AD.constant( math.cos( a() ) ) end,

}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- SCALARS

AD[SCALAR] = function(names)
  local result = table.pack( AD.symbol(names, SCALAR) )
  for i=1,#result do
    result[i].eval = function(self,values)
      return assert(values[self.name], "Undefined value " .. self.name)
    end
  end
  return table.unpack(result)
end

-- CONSTANT OPERATIONS

AD.op[SCALAR] = {
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = op('add', SCALAR, {a,b},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   local b = self.args[2]:eval(...)
		   return a + b
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
		 end)
    return s
  end,

  exp = function(a)
    local a = coercion(a)
    local s = op('exp', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.exp(a)
		 end)
    return s
  end,

  cos = function(a)
    local a = coercion(a)
    local s = op('cos', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.cos(a)
		 end)
    return s
  end,

  sin = function(a)
    local a = coercion(a)
    local s = op('sin', SCALAR, {a},
		 function(self, ...)
		   local a = self.args[1]:eval(...)
		   return math.sin(a)
		 end)
    return s
  end,

}
