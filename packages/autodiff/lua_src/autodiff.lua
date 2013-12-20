autodiff    = autodiff or {}
autodiff.op = autodiff.op or {}

------------------------------------------------------------------------------

-- dtype constants
local CONSTANT = 'constant'
local SCALAR   = 'scalar'
local MATRIX   = 'matrix'

------------------------------------------------------------------------------

-- auxiliar function which inserts gradient of a given symbol name
local function insert_grad(t, key, value)
  local t = t or {}
  t[key] = (not t[key] and value) or (t[key] + value)
  return t
end

-- all the symbols declared in a program will be stored here, so, symbol names
-- couldn't be reused
local SYMBOLS = {}

-- table for inference computation given a pair of symbol dtypes
local infer_table = {
  [CONSTANT] = { [CONSTANT]=CONSTANT, [SCALAR]=SCALAR, [MATRIX]=MATRIX },
  [SCALAR]   = { [CONSTANT]=SCALAR,   [SCALAR]=SCALAR, [MATRIX]=MATRIX },
  [MATRIX]   = { [CONSTANT]=MATRIX,   [SCALAR]=MATRIX, [MATRIX]=MATRIX },
}

-- function which inferes the dtype given a list of symbols
local function infer(...)
  local arg   = table.pack(...)
  local dtype = (type(arg[1]) == "number" and CONSTANT) or arg[1].dtype
  for i=2,#arg do
    local argi_dtype = (type(arg[i]) == "number" and CONSTANT) or arg[i].dtype
    dtype = infer_table[dtype][argi_dtype]
  end
  return dtype
end

-- a function which declares symbols given its name and its dtype. A symbol
-- has two basic methods:
--
-- e:eval(values,cache) => evaluates the expresion using the given variable
--                         values table and the given cache table. The cache
--                         table is useful to store partial computations,
--                         improving the efficiency of the computation.
--
-- e:diff(seed,result) => returns a table with all the possible gradients of the
--                        given symbol. The gradient table is indexed by symbol
--                        names. The given seed parameter is a value with the
--                        size of the output produced by the given symbolic
--                        function. The result table stores all the computed
--                        gradients, and is equivalent to the returned table.
local function symbol(name,dtype)
  local t
  if SYMBOLS[name] then
    t = SYMBOLS[name]
    assert(t.dtype == dtype, "Symbol redifinition is not allowed")
  else
    -- a new metatable for each symbol, allows to redefine the operations for
    -- specific symbol types
    local symbol_mt = {
      __add  = function(a,b) return autodiff.op[ infer(a,b) ].add(a,b) end,
      __sub  = function(a,b) return autodiff.op[ infer(a,b) ].sub(a,b) end,
      __mul  = function(a,b) return autodiff.op[ infer(a,b) ].mul(a,b) end,
      __div  = function(a,b) return autodiff.op[ infer(a,b) ].div(a,b) end,
      __unm  = function(a)   return autodiff.op[ infer(a) ].unm(a)     end,
      __pow  = function(a,b) return autodiff.op[ infer(a,b) ].pow(a,b) end,
      __tostring = function(s) return s.name end,
      __eq = function(a,b) return a.name == b.name end,
    }
    -- the symbol table
    t = {
      name     = name,
      dtype    = dtype,
      issymbol = true,
      -- basic eval function, returns the value stored at values table
      eval     = function(self,values)
	return assert(values[self.name], "Undefined value " .. self.name)
      end,
      -- default diff table, introduces the given seed at the result table
      diff     = function(self, seed, result)
	return insert_grad(result, self.name, seed)
      end,
      --
      last = nil,
      -- method for debug purposes
      to_dot_string = function(self,id,parent,edges)
	local aux = { string.format("%s [shape=box];", name) }
	if parent then
	  local edge_str = string.format("%s -> %s;", name, parent)
	  if not edges[edge_str] then
	    table.insert(aux, edge_str)
	    edges[edge_str] = true
	  end
	end
	return table.concat(aux, "\n")
      end,
    }
    -- stores the symbol at the SYMBOLS table
    SYMBOLS[name] = t
    --
    setmetatable(t, symbol_mt)
  end
  return t
end

-- auxiliary coercion function, converts Lua types in symbolic types
local function coercion(a)
  if type(a) == "number" then return autodiff.constant(a)
  else return a
  end
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- function for symbol names clear
function autodiff.clear()
  SYMBOLS = {}
end

-- removes a given list of names from the symbols table
function autodiff.remove(...)
  for i,name in ipairs(table.pack(...)) do
    SYMBOLS[name] = nil
  end
end

-- this functions adds a new operation with the given data
function autodiff.add_op(name, dtype, args, eval_func, diff_func)
  local s = symbol(string.format("(%s %s)", name,
				 iterator(ipairs(args)):select(2):
				 map(tostring):concat(" ")),
		   dtype)
  s.isop = name
  s.args = args
  -- eval function for operations
  s.eval = function(self, values, prev_cache)
    -- create the cache if not given
    local cache = prev_cache or {}
    local v = values[self.name] or cache[self.name] or eval_func(self, values, cache)
    -- store the last value in the cache and in the symbol object
    cache[self.name] = v
    self.last = v
    return v
  end
  -- diff function for operations
  s.diff = function(self, seed, result)
    -- by default the seed is a symbol as the symbol output filled with 1s
    local seed = seed or autodiff.op.fill(self,1)
    return diff_func(self, seed, insert_grad(result, self.name, seed))
  end
  -- auxiliary function for debugging purposes
  s.to_dot_string = function(self,id,parent,names,edges)
    local edges = edges or {}
    local names = names or {}
    local id  = id or { 0 }
    local aux = {}
    local name_str
    if names[self.name] then
      name_str = names[self.name]
    else
      name_str = "op" .. id[1]
      id[1] = id[1] + 1
      names[self.name] = name_str
      table.insert(aux, string.format('%s [label="%s"];', name_str, self.isop))
    end
    if parent then
      local edge_str = string.format("%s -> %s;", name_str, parent)
      if not edges[edge_str] then
	table.insert(aux, edge_str)
	edges[edge_str] = true
      end
    end
    for _,v in ipairs(self.args) do
      local str = v:to_dot_string(id,name_str,names,edges)
      table.insert(aux, str)
    end
    return table.concat(aux, "\n")
  end
  return s
end

-- helper name
local add_op = autodiff.add_op
--

-- Function for declaration of symbols from the exterior of autodiff program. It
-- receives an string with a list of names, and a dtype
function autodiff.symbol(names,dtype)
  local result = iterator(names:gmatch("[^%s]+")):
  map(function(name) return symbol(name,dtype) end):table()
  return table.unpack(result)
end

-- Function which converts a given table with symbols in a multi-evaluated
-- function. It receives an args table where the function arguments are stored
-- in order. The shared_values table stores pairs name,value which are shared
-- between symbolic expressions and your Lua program. The resulting function
-- will return many values as the number of symbols are given in s table.
function autodiff.func(s,args,shared_values,cache)
  assert(type(s) == "table")
  if s.issymbol then s = { s } end
  local args,shared_values = args or {},shared_values or {}
  for i,t in ipairs(args) do
    assert(type(t)=="table" and t.issymbol,
	   "Argument " .. i .. " is not a symbol")
  end
  for name,_ in pairs(shared_values) do
    assert(SYMBOLS[name], "Undefined symbol " .. name)
  end
  -- the returned function is a closure
  return function(...)
    local cache = cache or {}
    local args2 = table.pack(...)
    assert(#args == #args2,
	   string.format("Incorrect number of arguments, expected %d, found %d\n",
			 #args, #args2))
    local values = iterator(ipairs(args)):
    map(function(k,v) return v.name,args2[k] end):table()
    for k,v in pairs(shared_values) do values[k] = v end
    local ret = iterator(ipairs(s)):
    map(function(_,current) return current:eval(values,cache) end):table()
    return table.unpack(ret)
  end
end

-- Function for differentiation. It receives a symbol function f, a table with
-- target symbols for which you want to compute gradients, and an optional
-- initial seed. The function returns as many symbolic expressions as values
-- contains the symbols table. All the returned expressions are gradients.
function autodiff.diff(f, symbols, seed)
  assert(type(f) == "table" and f.issymbol)
  assert(type(symbols) == "table")
  if symbols.issymbol then symbols = { symbols } end
  local all_diff = f:diff(seed)
  return table.unpack(iterator(ipairs(symbols)):
		      map(function(_,s) return all_diff[s.name] end):
		      table())
end

-- auxiliary metatable for the autodiff.op table
setmetatable(autodiff.op,
	     {
	       __index = function(s,key)
		 return rawget(s,key) or
		   function(...)
		     assert(select('#',...) > 0,
			    "Incorrect number of arguments")
		     local dtype = infer(...)
		     local t = assert(autodiff.op[dtype],
				      "Incorrect type " .. dtype)
		     local t = assert(t[key],
				      "Operation: " .. key .. " not implemented for type: " .. dtype)
		     return t(...)
		   end
	       end,
	     })

-- auxiliary function for debugging purposes
function autodiff.dot_graph(s, filename)
  local f = io.open(filename, "w")
  f:write("digraph g {\n rankdir=BT;\n")
  f:write( s:to_dot_string() )
  f:write("}\n")
  f:close()
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- CONSTANTS

-- declaration of a constant symbol
autodiff.constant = function(...)
  local arg = table.pack(...)
  local result = {}
  for _,value in ipairs(arg) do
    local s = autodiff.symbol(tostring(value), CONSTANT)
    s.value = value
    s.eval  = function(self) return self.value end
    s.diff  = function(self, seed, result)
      local result = result or {}
      result[self.name] = autodiff.constant( 0 )
      return result
    end
    s.to_dot_string = function(self,id,parent,names,edges)
      local id  = id or { 0 }
      local aux = {}
      local name_str
      if names[self.name] then name_str = names[self.name]
      else
	name_str = "K" .. id[1]
	id[1] = id[1] + 1
	names[self.name] = name_str
	table.insert(aux, string.format('%s [shape=box,label="%s"];',
					name_str, self.name))
      end
      if parent then
	local edge_str = string.format("%s -> %s;", name_str, parent)
	if not edges[edge_str] then
	  table.insert(aux, edge_str)
	  edges[edge_str] = true
	end
      end
      return table.concat(aux, "\n")
    end
    table.insert(result, s)
  end
  return table.unpack(result)
end

-- CONSTANT OPERATIONS

autodiff.op[CONSTANT] = {
  
  -- the basic operations
  add = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a:eval() + b:eval() ) end,
  sub = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a:eval() - b:eval() ) end,
  pow = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a:eval() ^ b:eval() ) end,
  unm = function(a)   local a=coercion(a) return autodiff.constant( - a:eval() )     end,
  mul = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a:eval() * b:eval() ) end,
  div = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant( a:eval() / b:eval() ) end,
  
  -- extended math expressions
  log = function(a) local a=coercion(a) return autodiff.constant( math.log( a:eval() ) ) end,
  exp = function(a) local a=coercion(a) return autodiff.constant( math.exp( a:eval() ) ) end,
  sin = function(a) local a=coercion(a) return autodiff.constant( math.sin( a:eval() ) ) end,
  cos = function(a) local a=coercion(a) return autodiff.constant( math.cos( a:eval() ) ) end,
  tan = function(a) local a=coercion(a) return autodiff.constant( math.tan( a:eval() ) ) end,
  sinh = function(a) local a=coercion(a) return autodiff.constant( math.sinh( a:eval() ) ) end,
  cosh = function(a) local a=coercion(a) return autodiff.constant( math.cosh( a:eval() ) ) end,
  tanh = function(a) local a=coercion(a) return autodiff.constant( math.tanh( a:eval() ) ) end,
  asin = function(a) local a=coercion(a) return autodiff.constant( math.asin( a:eval() ) ) end,
  acos = function(a) local a=coercion(a) return autodiff.constant( math.acos( a:eval() ) ) end,
  atan = function(a) local a=coercion(a) return autodiff.constant( math.atan( a:eval() ) ) end,
  asinh = function(a) local a=coercion(a) return autodiff.constant( math.asinh( a:eval() ) ) end,
  acosh = function(a) local a=coercion(a) return autodiff.constant( math.acosh( a:eval() ) ) end,
  atanh = function(a) local a=coercion(a) return autodiff.constant( math.atanh( a:eval() ) ) end,

  -- matrix expressions
  transpose = function(a) local a=coercion(a) return autodiff.constant( a:eval() ) end,
  fill = function(a,b) local a,b=coercion(a),coercion(b) return autodiff.constant(b) end,
}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- SCALARS

autodiff[SCALAR] = function(names) return autodiff.symbol(names, SCALAR) end

-- SCALAR OPERATIONS

autodiff.op[SCALAR] = {
  
  -- basic math operations
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifactions
    if a == autodiff.constant(0) then return b
    elseif b == autodiff.constant(0) then return a
    end
    --
    local s = add_op('+', SCALAR, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       return a + b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(seed, result)
		       b:diff(seed, result)
		       return result
		     end)
    return s
  end,
  
  sub = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a + (-1 * b)
  end,

  mul = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifactions
    if a == autodiff.constant(0) or b == autodiff.constant(0) then
      return autodiff.constant(0)
    elseif a == autodiff.constant(1) then return b
    elseif b == autodiff.constant(1) then return a
    end
    --
    local s = add_op('*', SCALAR, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       return a * b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(seed*b, result)
		       b:diff(seed*a, result)
		       return result
		     end)
    return s
  end,
  
  div = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a * (b^(-1))
  end,

  pow = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifactions
    if a == autodiff.constant(0) then
      return autodiff.constant(0)
    elseif b == autodiff.constant(0) then return autodiff.constant(1)
    elseif b == autodiff.constant(1) then return a
    end
    --
    local s = add_op('^', SCALAR, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       return a^b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(b * (a^(b-1)) * seed, result)
		       return result
		     end)
    return s
  end,
  
  unm = function(a) local a = coercion(a) return (-1) * a end,
  
  -- extended math operations
  
  log = function(a)
    local a = coercion(a)
    local s = add_op('log', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.log(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(1/a * seed, result)
		       return result
		     end)
    return s
  end,

  exp = function(a)
    local a = coercion(a)
    local s = add_op('exp', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.exp(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(self * seed, result)
		       return result
		     end)
    return s
  end,

  sin = function(a)
    local a = coercion(a)
    local s = add_op('sin', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.sin(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cos(a) * seed, result)
		       return result
		     end)
    return s
  end,

  cos = function(a)
    local a = coercion(a)
    local s = add_op('cos', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.cos(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(-autodiff.op.sin(a) * seed, result)
		       return result
		     end)
    return s
  end,

  sinh = function(a)
    local a = coercion(a)
    local s = add_op('sinh', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.sinh(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cosh(a) * seed, result)
		       return result
		     end)
    return s
  end,

  cosh = function(a)
    local a = coercion(a)
    local s = add_op('cosh', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.cosh(a)
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(autodiff.op.sinh(a) * seed, result)
		       return result
		     end)
    return s
  end,

  tanh = function(a)
    local a = coercion(a)
    local s = add_op('tanh', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.tanh(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff((1 - self^2) * seed, result)
		       return result
		     end)
    return s
  end,

  -- matrix operations
  fill = function(a,b) return b end,
  transpose = function(a) return a end,

}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- MATRIXS

autodiff[MATRIX] = function(names)
  local t = table.pack(autodiff.symbol(names, MATRIX))
  for i=1,#t do
    t[i].diff = function(self, seed, result)
      return insert_grad(result, self.name, seed)
    end
  end
  return table.unpack(t)
end

-- CONSTANT OPERATIONS

autodiff.op[MATRIX] = {
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = add_op('+', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       -- simplifications
		       if a == 0 then return b
		       elseif b == 0 then return a
		       end
		       --
		       return a + b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(seed, result)
		       b:diff(seed, result)
		       return result
		     end)
    return s
  end,
  
  sub = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a + (-1 * b)
  end,

  mul = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = add_op('*', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       -- simplifications
		       if a == 0 or b == 0 then return 0
		       elseif a == 1 then return b
		       elseif b == 1 then return a
		       end
		       --
		       return a * b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(seed*b, result)
		       b:diff(autodiff.op.transpose(a)*seed, result)
		       return result
		     end)
    return s
  end,
  
  div = function(a,b)
    local a,b = coercion(a),coercion(b)
    assert(a.dtype ~= MATRIX or b.dtype ~= MATRIX,
	   "Incorrect types, div between MATRIX and MATRIX is not implemented")
    return a * (b^(-1))
  end,

  pow = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = add_op('^', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       -- sanity check
		       assert(type(a) == "matrix")
		       assert(type(b) ~= "matrix",
			      "Impossible to compute pow with a 2nd matrix argument")
		       -- simplifications
		       if     b == 0 then return matrix.as(a):ones()
		       elseif b == 1 then return a
		       end
		       --
		       return a:clone():pow(b)
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       local seed = autodiff.op.cmul(seed, b*a^(b-1))
		       a:diff(seed, result)
		       return result
		     end)
    return s
  end,
  
  unm = function(a)
    local a = coercion(a)
    return (-1) * a
  end,
  
  log = function(a)
    local a = coercion(a)
    local s = add_op('log', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():log()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       local da = a:diff(seed, result)
		       return autodiff.op.cmul(autodiff.op.pow(a, -1), da)
		     end)
    return s
  end,

  exp = function(a)
    local a = coercion(a)
    local s = add_op('exp', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():exp()
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(autodiff.op.cmul(self, seed), result)
		       return result
		     end)
    return s
  end,

  cos = function(a)
    local a = coercion(a)
    local s = add_op('cos', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():cos()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cmul(-autodiff.op.sin(a), seed), result)
		       return result
		     end)
    return s
  end,

  sin = function(a)
    local a = coercion(a)
    local s = add_op('sin', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():sin()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cmul(autodiff.op.cos(a), seed), result)
		       return result
		     end)
    return s
  end,

  transpose = function(a)
    local a = coercion(a)
    local s = add_op('T', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:transpose()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.transpose(seed), result)
		       return result
		     end)
    return s
  end,

  cmul = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = add_op('.*', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       if a == 0 or b == 0 then return 0 end
		       if type(a) == "number" or type(b) == "number" then
			 return a*b
		       end
		       return a:cmul(b)
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(autodiff.op.cmul(a,seed), result)
		       b:diff(autodiff.op.cmul(b,seed), result)
		       return result
		     end)
    return s
  end,
  
  fill = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = add_op('fill', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       assert(type(a) == "matrix")
		       assert(type(b) == "number")
		       return matrix.as(a):fill(b)
		     end,
		     function(self, seed, result)
		       return result
		     end)
    return s
  end,
  
  sum = function(a,b)
    local a,b = coercion(a),b and coercion(b)
    local s = add_op('sum', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2] and self.args[2]:eval(...)
		       assert(type(a) == "matrix")
		       assert(not b or type(b)=="number")
		       return a:sum(b)
		     end,
		     function(self, seed, result)
		       error("NOT IMPLEMENTED")
		     end)
    return s
  end,
}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
