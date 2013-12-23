autodiff    = autodiff or {}
autodiff.op = autodiff.op or {}

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- auxiliary functions for variable name generation
local gen_arg_name
local reset_var_id
do
  local var_name_id=0
  reset_var_id = function() var_name_id = 0 end
  gen_var_name = function()
    var_name_id=var_name_id+1
    return string.format("__var%d__", var_name_id)
  end
end

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- COMPILER OUT: Lua CLASS, developed from scratch, not using April-ANN
-- class. Allows to write the compilation output to a file

local compiler_out = {}
local compiler_out_mt = {}

-- constructor
setmetatable(compiler_out,
	     {
	       __call = function(self,filename)
		 local f = io.open(filename, "w") or error("Impossible to open: ".. filename)
		 local obj = { f=f, indent=1, active_vars={}, cache_counts={} }
		 setmetatable(obj,compiler_out_mt)
		 obj.f:write("return function(arg,cache)\n")
		 return obj
	       end,
	     })

-- methods
compiler_out_mt.__index = {
  -- basic methods
  write_indent = function(self)
    local tbl = {}
    for i=1,self.indent do table.insert(tbl, "  ") end
    self.f:write(table.concat(tbl, ""))
  end,
  write_return = function(self, var_name)
    self:write_indent()
    self.f:write(string.format("return %s\n", var_name))
  end,
  close = function(self)
    self.f:write("end\n")
    self.f:close()
  end,
  new_function = function(self)
    self.f:write("end,\n")
    self.f:write("function(arg,cache)\n")
    self.active_vars = {}
  end,
  count_cache = function(self,var_name)
    self.cache_counts[var_name] = (self.cache_counts[var_name] or 0) + 1
  end,
  get_cache_count = function(self, var_name)
    return self.cache_counts[var_name] or 0
  end,
  -- variable declaration methods
  write_var = function(self,var_name)
    if not self.active_vars[var_name] then
      self:write_indent()
      self.f:write(string.format("local %s\n", var_name))
    end
    self.active_vars[var_name] = true
  end,
  write_initial_var = function(self,var_name,name)
    if not self.active_vars[var_name] then
      self:write_indent()
      self.f:write(string.format("local %s = arg[%q]\n", var_name, name))
    end
    self.active_vars[var_name] = true
  end,
  write_initial_constant = function(self,var_name,value)
    if not self.active_vars[var_name] then
      self:write_indent()
      self.f:write(string.format("local %s = %s\n",
				 var_name, tostring(value)))
    end
    self.active_vars[var_name] = true
  end,
  -- expression methods
  begin_expression = function(self, var_name, parent_count)
    -- The same parent and children cache count means that they are dependent,
    -- so the children always come with the same parent. If the counts are
    -- different, then the children has a dependence in paths different than the
    -- current parent. In the first case, it is not necessary to introduce a new
    -- block, because children and parent always come together. In the second
    -- case, a block with a cache check is needed.
    if self.cache_counts[var_name] > (parent_count or 0) then
      self:write_indent()
      self.f:write(string.format("if not cache[%q] then\n", var_name))
      self.indent = self.indent + 1
    end
  end,
  write_expr_line = function(self, expression)
    self:write_indent()
    self.f:write(string.format("%s\n", expression))
  end,
  write_expr_assign = function(self, var_name, expression)
    self:write_indent()
    if not self.active_vars[var_name] then
      self.f:write("local ")
    end
    self.f:write(string.format("%s = (%s)\n", var_name, expression))
    self.active_vars[var_name] = true
  end,
  end_expression = function(self, var_name, parent_count)
    assert(self.active_vars[var_name],
	   "Declare expresion vars before writing them")
    -- The same parent and children cache count means that they are dependent,
    -- so the children always come with the same parent. If the counts are
    -- different, then the children has a dependence in paths different than the
    -- current parent. In the first case, it is not necessary to introduce a new
    -- block, because children and parent always come together. In the second
    -- case, a block with a cache check is needed.
    if self.cache_counts[var_name] > (parent_count or 0) then
      self:write_indent()
      self.f:write(string.format("cache[%q] = %s\n", var_name, var_name))
      self.indent = self.indent - 1
      self:write_indent()
      self.f:write(string.format("else -- if not cache[%q]\n", var_name))
      self.indent = self.indent + 1
      self:write_indent()
      self.f:write(string.format("%s = cache[%q]\n", var_name, var_name))
      self.indent = self.indent - 1
      self:write_indent()
      self.f:write(string.format("end -- if not cache[%q] else ... end\n",
	  var_name))
    end
  end,
}
------------------------------------------------------------------------------

-- dtype constants
local CONSTANT = 'constant'
local SCALAR   = 'scalar'
local MATRIX   = 'matrix'
local TABLE    = 'table'
local STRING   = 'string'

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- auxiliar function which inserts gradient of a given symbol name, accumulating
-- gradients which come from different graph paths
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
  [CONSTANT] = { [CONSTANT]=CONSTANT, [SCALAR]=SCALAR, [MATRIX]=MATRIX, [TABLE]=CONSTANT, [STRING]=CONSTANT },
  [SCALAR]   = { [CONSTANT]=SCALAR,   [SCALAR]=SCALAR, [MATRIX]=MATRIX, [TABLE]=SCALAR,   [STRING]=SCALAR   },
  [MATRIX]   = { [CONSTANT]=MATRIX,   [SCALAR]=MATRIX, [MATRIX]=MATRIX, [TABLE]=MATRIX,   [STRING]=MATRIX   },
  [TABLE]    = { [CONSTANT]=CONSTANT, [SCALAR]=SCALAR, [MATRIX]=MATRIX, [TABLE]=TABLE,    [STRING]=TABLE    },
  [STRING]   = { [CONSTANT]=CONSTANT, [SCALAR]=SCALAR, [MATRIX]=MATRIX, [TABLE]=TABLE,    [STRING]=STRING   },
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
--
-- e:compile(compiler_out) => produces code for the evaluation of the object
--                            and writes it to the given compiler_out instance.
local function symbol(name,dtype)
  local t
  if SYMBOLS[name] then
    t = SYMBOLS[name]
    assert(t.dtype == dtype, "Symbol redifinition is not allowed: " .. name)
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
      __lt = function(a,b) return a.name <  b.name end,
      __le = function(a,b) return a.name <= b.name end,
    }
    -- the symbol table
    t = {
      name     = name,   -- the name identifies the symbol
      dtype    = dtype,  -- indicates the expected type of the symbol
      issymbol = true,   -- indicates that this table is a symbol
      dims     = nil,    -- it is possible to write the dimensions if needed the
      -- following method removes the var_name associated with the compilation
      -- of the symbol
      clear_var_name = function(self) self.var_name = nil end,
      -- modifies the dimensions of the symbol shape
      set_dims = function(self,...)
	self.dims = table.pack(...)
	if type(self.dims[1]) == "table" then
	  assert(#self.dims == 1,
		 "set_dims accepts ONE table or a MULTIPLE numbers list")
	  self.dims = self.dims[1]
	end
      end,
      -- default eval function, returns the value stored at values table
      eval = function(self,values)
	local m = values[self.name] or error("Undefined value " .. self.name)
	return m
      end,
      -- default diff function, introduces the given seed at the result table
      diff = function(self, seed, result)
	return insert_grad(result, self.name, seed)
      end,
      -- default compile function, reserves a var_name if needed, and writes the
      -- initialization of the variable
      compile = function(self,dest)
	if not self.var_name then
	  self.var_name = gen_var_name()
	end
	dest:write_initial_var(self.var_name,self.name)
      end,
      -- the last value of eval function is stored here
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
    -- the symbol is stored at the SYMBOLS table
    SYMBOLS[name] = t
    --
    setmetatable(t, symbol_mt)
  end
  return t
end

-- auxiliary coercion function, converts Lua types in symbolic types
local function coercion(a)
  if type(a) == "number" then return autodiff.constant(a)
  elseif type(a) == "table" and not a.issymbol then return autodiff.table(a)
  elseif type(a) == "string" then return autodiff.string(a)
  else return a
  end
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- function for symbol names clear, it is useful when you want to compile
-- totally different functions, which don't share symbols
function autodiff.clear()
  SYMBOLS = {}
end

-- removes a given list of names from the symbols table
function autodiff.remove(...)
  for i,name in ipairs(table.pack(...)) do
    SYMBOLS[name] = nil
  end
end

-- this functions returns a new operation with the given data
function autodiff.gen_op(name, dtype, args, eval_func, diff_func, compile)
  local compile = compile or function() error("COMPILATION NOT IMPLEMENTED") end
  -- an operation is a symbol with the given type, and with a name which is a
  -- concatenation of its arguments
  local s = symbol(string.format("(%s %s)", name,
				 iterator(ipairs(args)):select(2):
				 map(tostring):concat(" ")),
		   dtype)
  -- this flag allows to distinguish between operations and standard symbols
  s.isop = name
  -- the arguments of the operation
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
    -- by default the seed is a symbol which has the type and shape of the
    -- operation output, but filled with 1s
    local seed = seed or autodiff.op.fill(self,1)
    return diff_func(self, seed, insert_grad(result, self.name, seed))
  end
  -- compilation function, compiles the operation
  s.compile = function(self,dest,parent_count)
    if not self.var_name then
      self.var_name = gen_var_name()
    end
    dest:begin_expression(self.var_name, parent_count)
    -- compiles the arguments list
    iterator(ipairs(self.args)):select(2):
    call('compile',dest,dest:get_cache_count(self.var_name)):apply()
    -- compiles the operation expression itself
    compile(self, dest)
    dest:end_expression(self.var_name, parent_count)
  end
  -- removes the associated var_name, and calls the clear_var_name of its
  -- arguments
  s.clear_var_name = function(self)
    self.var_name = nil
    iterator(ipairs(self.args)):select(2):call('clear_var_name'):apply()
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
local gen_op = autodiff.gen_op
--

-- Function for declaration of symbols from the exterior of autodiff program. It
-- receives an string with a list of names, and a dtype
function autodiff.symbol(names,dtype)
  local result = iterator(names:gmatch("[^%s]+")):
  map(function(name) return symbol(name,dtype) end):table()
  return table.unpack(result)
end

-- Function which compiles any number of symbols (in a table) into a
-- multi-evaluated function. It receives an args table where the function
-- arguments are stored in the expected order. The shared_values table stores
-- pairs name,value which are shared between symbolic expressions and your Lua
-- program. The resulting function will return as many values as the number of
-- symbols are given in s table.
function autodiff.func(s, args, shared_values)
  assert(type(s) == "table")
  if s.issymbol then s = { s } end
  -- removes the stored var_names
  iterator(ipairs(s)):select(2):call('clear_var_name'):apply()
  -- checks the args table, and builds a dictionary for check that all the
  -- necessary symbols has an argument or a shared_value
  local symbols_dict = {}
  local args,shared_values = args or {},shared_values or {}
  for i,t in ipairs(args) do
    assert(type(t)=="table" and t.issymbol and SYMBOLS[t.name],
	   "Argument " .. i .. " is not a symbol or it was cleared")
    assert(not symbols_dict[t.name],
	   "An argument was given two times: " .. t.name)
    symbols_dict[t.name] = true
  end
  -- checks the shared_values table
  for name,_ in pairs(shared_values) do
    assert(SYMBOLS[name], "Undefined or cleared symbol " .. name)
    assert(not symbols_dict[name],
	   "An argument or shared_value was given two times: " .. name)
    symbols_dict[name] = true
  end
  -- COMPILATION PROCEDURE
  reset_var_id() -- resets the variable id counter
  local filename = os.tmpname()
  local dest = compiler_out(filename)
  -- FIRST, traverse the symbols to acquire cache counts, which will be used to
  -- optimize the produced code, and checks if all the not op symbol variables
  -- are given as argument or as shared_value (symbols_dict)
  function count_cache(v,dest)
    if not v.var_name then v.var_name = gen_var_name() end
    dest:count_cache(v.var_name)
    if v.isop then
      for _,v2 in ipairs(v.args) do count_cache(v2,dest) end
    else assert(v.dtype==CONSTANT or symbols_dict[v.name],
		"Symbol not found as argument or shared_variable: ".. v.name)
    end
  end
  -- count over all the given symbols
  for i,current_s in ipairs(s) do count_cache(current_s,dest) end
  -- SECOND, traverse the symbols producing the source code
  for i,current_s in ipairs(s) do
    if i>1 then dest:new_function() end
    if current_s.isop then
      -- declare local vars at the beginning of the function
      function get_vars(v,dest)
	if not v.isop then v:compile(dest)
	else
	  if not v.var_name then v.var_name = gen_var_name() end
	  dest:write_var(v.var_name)
	  for _,v2 in ipairs(v.args) do get_vars(v2,dest) end
	end
      end
      get_vars(current_s,dest)
    end
    -- compile the symbol
    current_s:compile(dest)
    dest:write_return(current_s.var_name)
  end
  dest:close()
  -- loading of the compiled program
  local funcs = table.pack(dofile(filename))
  local f = io.open(filename, "r")
  local program = f:read("*a")
  f:close()
  os.remove(filename)
  -- the returned function is a closure which inserts the input arguments and
  -- shared_values in a dictionary; this dictionary is passed to each of the
  -- previously compiled functions
  return function(...)
    local args2 = table.pack(...)
    assert(#args == #args2,
	   string.format("Incorrect number of arguments, expected %d, found %d\n",
			 #args, #args2))
    local values = iterator(ipairs(args)):
    map(function(k,v) return v.name,args2[k] end):table()
    for k,v in pairs(shared_values) do values[k] = v end
    local cache = {}
    local ret = iterator(ipairs(funcs)):select(2):
    map(function(f) return f(values,cache) end):table()
    return table.unpack(ret)
  end,program
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
  f:write("digraph g {\nrankdir=BT;\n")
  f:write( s:to_dot_string() )
  f:write("}\n")
  f:close()
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- CONSTANT

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
    s.compile = function(self,dest)
      if not self.var_name then
	self.var_name = gen_var_name()
      end
      dest:write_initial_constant(self.var_name,self:eval())
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
  abs = function(a) local a=coercion(a) return autodiff.constant( math.abs(a:eval() )) end,
  log = function(a) local a=coercion(a) return autodiff.constant( math.log( a:eval() ) ) end,
  exp = function(a) local a=coercion(a) return autodiff.constant( math.exp( a:eval() ) ) end,
  --
  sin = function(a) local a=coercion(a) return autodiff.constant( math.sin( a:eval() ) ) end,
  cos = function(a) local a=coercion(a) return autodiff.constant( math.cos( a:eval() ) ) end,
  tan = function(a) local a=coercion(a) return autodiff.constant( math.tan( a:eval() ) ) end,
  --
  sinh = function(a) local a=coercion(a) return autodiff.constant( math.sinh( a:eval() ) ) end,
  cosh = function(a) local a=coercion(a) return autodiff.constant( math.cosh( a:eval() ) ) end,
  tanh = function(a) local a=coercion(a) return autodiff.constant( math.tanh( a:eval() ) ) end,
  --
  asin = function(a) local a=coercion(a) return autodiff.constant( math.asin( a:eval() ) ) end,
  acos = function(a) local a=coercion(a) return autodiff.constant( math.acos( a:eval() ) ) end,
  atan = function(a) local a=coercion(a) return autodiff.constant( math.atan( a:eval() ) ) end,
  --
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

-- SCALAR

autodiff[SCALAR] = function(names) return autodiff.symbol(names, SCALAR) end

-- SCALAR OPERATIONS

autodiff.op[SCALAR] = {
  
  -- basic math operations
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifications
    if b<a then a,b=b,a end
    if a == autodiff.constant(0) then return b
    elseif b == autodiff.constant(0) then return a
    end
    --
    local s = gen_op('+', SCALAR, {a,b},
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
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, "+", b.var_name }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
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
    if b<a then a,b=b,a end
    if a == autodiff.constant(0) or b == autodiff.constant(0) then
      return autodiff.constant(0)
    elseif a == autodiff.constant(1) then return b
    elseif b == autodiff.constant(1) then return a
    end
    --
    local s = gen_op('*', SCALAR, {a,b},
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
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, "*", b.var_name }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
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
    local s = gen_op('^', SCALAR, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       return a^b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(b * (a^(b-1)) * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, "^", b.var_name }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,
  
  unm = function(a) local a = coercion(a) return (-1) * a end,
  
  -- extended math operations
  
  log = function(a)
    local a = coercion(a)
    local s = gen_op('log', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.log(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(1/a * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.log(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  exp = function(a)
    local a = coercion(a)
    local s = gen_op('exp', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.exp(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(self * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.exp(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  sin = function(a)
    local a = coercion(a)
    local s = gen_op('sin', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.sin(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cos(a) * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.sin(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  cos = function(a)
    local a = coercion(a)
    local s = gen_op('cos', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.cos(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(-autodiff.op.sin(a) * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.cos(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  sinh = function(a)
    local a = coercion(a)
    local s = gen_op('sinh', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.sinh(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cosh(a) * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.sinh(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  cosh = function(a)
    local a = coercion(a)
    local s = gen_op('cosh', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.cosh(a)
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(autodiff.op.sinh(a) * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.cosh(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  tanh = function(a)
    local a = coercion(a)
    local s = gen_op('tanh', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.tanh(a)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff((1 - self^2) * seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.tanh(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    return s
  end,

  abs = function(a)
    local a = coercion(a)
    local s = gen_op('abs', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return math.abs(a)
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(seed/self, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { "math.abs(", a.var_name, ")" }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
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

-- MATRIX

function check_dims(a,b)
  if a and b then
    if #a ~= #b then return false end
    for i=1,#a do
      if a[i] ~= b[i] then return false end
    end
  end
  return true
end

autodiff[MATRIX] = function(names)
  local t = table.pack(autodiff.symbol(names, MATRIX))
  for i=1,#t do
    t[i].diff = function(self, seed, result)
      return insert_grad(result, self.name, seed)
    end
    local old_eval = t[i].eval
    t[i].eval = function(self,values)
      local m = old_eval(self,values)
      assert( check_dims(t[i].dims, m:dim()),
	      "Incorrect dimensions, expected %s, found %s",
	      table.concat(t[i].dims or {}, "x"),
	      table.concat(m:dim(), "x") )
      return m
    end
    local mt = getmetatable(t[i])
    mt.__call = function(t, ...) return op.slice(t, ...) end
  end
  return table.unpack(t)
end

-- MATRIX OPERATIONS

-- local function broadcast(a,b)
--   local dims_a,dims_b = a.dims,b.dims
--   assert(#dims_a == #dims_b, "Incorrect dimension sizes")
--   local i=1 while i<=#dims_a and dims_a[i] == dims_b[i] do i=i+1 end
--   if i<=#dims_a then
--     local source
--     local dest
--     local tgt
--     if dims_a[i] == 1 then
--       tgt,source,dest = dims_a,a,autodiff.op.fill(b,0)
--       a = dest
--     elseif dims_b[i] == 1 then
--       tgt,source,dest = dims_b,b,autodiff.op.fill(a,0)
--       b = dest
--     else
--       error("Not aligned dimensions")
--     end
--     while i<=#dims_a do
--       assert(dims_a[i] == dims_b[i] or tgt[i] == 1,
-- 	     "Trailing dimensions do not match")
--       i=i+1
--     end
--     autodiff.op.slide(dest, {size=tgt, step=tgt},
-- 		      function(sw) return autodiff.op.copy(sw,source) end)

--   end
--   return a,b
-- end

autodiff.op[MATRIX] = {
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifcations
    if b<a then a,b=b,a end
    if a == autodiff.constant(0) then return b
    elseif b == autodiff.constant(0) then return a
    end
    --
    local s = gen_op('+', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       -- simplifications
		       if a == 0 then return b
		       elseif b == 0 then return a
		       end
		       return a + b
		     end,
		     function(self, seed, result)
		       local a,b = self.args[1],self.args[2]
		       a:diff(seed, result)
		       b:diff(seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, '+', b.var_name }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    if a.dims or b.dims then
      assert( check_dims(a.dims, b.dims),
	      "Incorrect dimensions")
      s:set_dims(a.dims or b.dims)
    end
    return s
  end,
  
  sub = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a + (-1 * b)
  end,

  mul = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifcations
    if a == autodiff.constant(0) or b == autodiff.constant(0) then
      return autodiff.constant(0)
    elseif a == autodiff.constant(1) then return b
    elseif b == autodiff.constant(1) then return a
    end
    --
    local s = gen_op('*', MATRIX, {a,b},
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
		       a:diff(seed*autodiff.op.transpose(b), result)
		       b:diff(autodiff.op.transpose(a)*seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, '*', b.var_name }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    if a.dims and b.dims then
      assert(#a.dims == 2 and #a.dims == #b.dims, "Incorrect dimensions")
      assert(a.dims[2] == b.dims[1],
	     string.format("Incorrect matrix dims for multiplication: %s * %s",
			   table.concat(a.dims, "x"),
			   table.concat(b.dims, "x")))
      s:set_dims(a.dims[1], b.dims[2])
    elseif a.dtype == CONSTANT or a.dtype == SCALAR then
      if b.dims then s:set_dims(b.dims) end
    elseif b.dtype == CONSTANT or b.dtype == SCALAR then
      if a.dims then s:set_dims(a.dims) end
    end
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
    -- simplifcations
    if a == autodiff.constant(0) then return autodiff.constant(0)
    elseif a == autodiff.constant(1) or b == autodiff.constant(0) then
      return autodiff.constant(1)
    end
    --
    local s = gen_op('^', MATRIX, {a,b},
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
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, '^', b.var_name }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, " "))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,
  
  unm = function(a)
    local a = coercion(a)
    return (-1) * a
  end,
  
  log = function(a)
    local a = coercion(a)
    local s = gen_op('log', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():log()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       local da = a:diff(seed, result)
		       return autodiff.op.cmul(autodiff.op.pow(a, -1), da)
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():log()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  exp = function(a)
    local a = coercion(a)
    local s = gen_op('exp', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():exp()
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(autodiff.op.cmul(self, seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():exp()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  cos = function(a)
    local a = coercion(a)
    local s = gen_op('cos', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():cos()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cmul(-autodiff.op.sin(a), seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():cos()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  sin = function(a)
    local a = coercion(a)
    local s = gen_op('sin', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():sin()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cmul(autodiff.op.cos(a), seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():sin()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  tanh = function(a)
    local a = coercion(a)
    local s = gen_op('tanh', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:clone():tanh()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.cmul(1 - self^2, seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():tanh()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  transpose = function(a)
    local a = coercion(a)
    local s = gen_op('T', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       return a:transpose()
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       a:diff(autodiff.op.transpose(seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':transpose()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then
      s:set_dims(iterator(ipairs(a.dims)):select(2):
		 reduce(function(acc,v) return table.insert(acc,1,v) end, {}))
    end
    return s
  end,

  cmul = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = gen_op('.*', MATRIX, {a,b},
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
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { a.var_name, ':cmul(', b.var_name, ')' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims or b.dims then
      assert( check_dims(a.dims, b.dims),
	      "Incorrect dimensions" )
      s:set_dims(a.dims or b.dims)
    end
    return s
  end,
  
  fill = function(a,b)
    local a,b = coercion(a),coercion(b)
    local s = gen_op('fill', MATRIX, {a,b},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       local b = self.args[2]:eval(...)
		       assert(type(a) == "matrix")
		       assert(type(b) == "number")
		       return matrix.as(a):fill(b)
		     end,
		     function(self, seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl = { 'matrix.as(', a.var_name, '):fill(', b.var_name, ')' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,
  
  sum = function(a)
    local a = coercion(a)
    local s = gen_op('sum', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return a:sum()
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(autodiff.op.fill(a, seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':sum()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    return s
  end,

  select = function(a,dim,value)
    local a,dim,value = coercion(a),coercion(dim),coercion(value)
    local s = gen_op('select', MATRIX, {a,dim,value},
		     function(self, ...)
		       local a     = self.args[1]:eval(...)
		       local dim   = self.args[2]:eval(...)
		       local value = self.args[3]:eval(...)
		       assert(type(a) == "matrix")
		       return a:select(dim,value)
		     end,
		     function(self, seed, result)
		       error("NOT IMPLEMENTED")
		     end,
		     function(self, dest)
		       local a     = self.args[1]
		       local dim   = self.args[2]
		       local value = self.args[3]
		       local str_tbl = { a.var_name, ':select(', dim.var_name, ',', value.var_name, ')' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    -- TODO: modify dims
    return s
  end,

  slice = function(...)
    local arg = iterator(ipairs(table.pack(...))):select(2):map(coercion):table()
    local s = gen_op('slice', MATRIX, arg,
		     function(self, ...)
		       local arg = iterator(ipairs(self.args)):
		       select(2):call('eval',...):table()
		       local a = table.remove(arg,1)
		       assert(type(a) == "matrix")
		       return a(table.unpack(arg))
		     end,
		     function(self, seed, result)
		       local a   = self.args[1]
		       local arg = iterator(range(2,#self.args)):
		       map(function(i) return self.args[i] end):table()
		       --
		       local dest = autodiff.op.fill(self,0)
		       dest = autodiff.op.copy(dest, a, table.unpack(arg))
		       a:diff(dest, result)
		       return result
		     end,
		     function(self, dest)
		       local a       = self.args[1]
		       local str_tbl = { a.var_name, '(' }
		       table.insert(str_tbl, self.args[2].var_name)
		       for i=3,#self.args do
			 table.insert(str_tbl, ',')
			 table.insert(str_tbl, self.args[i].var_name)
		       end
		       table.insert(str_tbl, ')')
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    -- TODO: modify dims
    return s
  end,

  copy = function(...)
    local arg = iterator(ipairs(table.pack(...))):select(2):map(coercion):table()
    local s = gen_op('copy', MATRIX, arg,
		     function(self, ...)
		       local arg = iterator(ipairs(self.args)):
		       select(2):call('eval',...):table()
		       local a = table.remove(arg,1)
		       local b = table.remove(arg,2)
		       assert(type(a) == "matrix")
		       assert(type(b) == "matrix")
		       a = a:clone()
		       if #arg > 0 then
			 a(table.unpack(arg)):copy(b)
		       else a:copy(b)
		       end
		       return a
		     end,
		     function(self, seed, result) return result end,
		     function(self, dest)
		       local a = self.args[1]
		       local b = self.args[2]
		       local str_tbl = { a.var_name, ':clone()' }
		       if #self.args > 2 then
			 table.insert(str_tbl, '(')
			 table.insert(str_tbl, self.args[3].var_name)
			 for i=4,#self.args do
			   table.insert(str_tbl, ',')
			   table.insert(str_tbl, self.args[i].var_name)
			 end
			 table.insert(str_tbl, ')')
		       end
		       table.insert(str_tbl, ':copy(')
		       table.insert(str_tbl, b.var_name)
		       table.insert(str_tbl, ')')
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    -- TODO: modify dims
    return s
  end,

  get = function(...)
    local arg = iterator(ipairs(table.pack(...))):select(2):map(coercion):table()
    local s = gen_op('get', SCALAR, arg,
		     function(self, ...)
		       local arg = iterator(ipairs(self.args)):
		       select(2):call('eval', ...):table()
		       local a = table.remove(arg,1)
		       assert(type(a) == "matrix")
		       return a:get(table.unpack(arg))
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       local arg = iterator(range(2,#self.args)):
		       map(function(i) return self.args[i] end):table()
		       local result = autodiff.op.fill(a, 0)
		       result = autodiff.op.copy(result,seed,table.unpack(arg))
		       a:diff(seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':get(' }
		       table.insert(str_tbl, self.args[2].var_name)
		       for i=3,#self.args do
			 table.insert(str_tbl, ',')
			 table.insert(str_tbl, self.args[i].var_name)
		       end
		       table.insert(str_tbl, ')')
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    return s
  end,

  abs = function(a)
    local a = coercion(a)
    local s = gen_op('abs', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return a:clone():abs()
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       a:diff(autodiff.op.cmul(seed,1/self), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():abs()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,
  
  logistic = function(a)
    local a = coercion(a)
    local s = gen_op('logistic', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return a:clone():scal(-1):exp():scalar_add(1):div(1)
		     end,
		     function(self, seed, result)
		       local a  = self.args[1]
		       local da = autodiff.op.cmul(a, 1-a)
		       a:diff(autodiff.op.cmul(da,seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name,
					 ':clone()',
					 ':scal(-1)',
					 ':exp()',
					 ':scalar_add(1)',
					 ':div(1)' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,
}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- TABLE

-- the table data type exists to allow passing Lua tables to the operators

autodiff[TABLE] = function(tbl)
  assert(type(tbl) == "table")
  local t = autodiff.symbol(table.tostring(tbl), TABLE)
  t.value = tbl
  t.eval  = function(self) return self.value end
  t.diff  = function(self, seed, result) return result end
  return t
end

-- TABLE OPERATIONS

autodiff.op[TABLE] = {}

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- STRING

-- the string data type exists to allow passing Lua strings to the operators

autodiff[STRING] = function(strl)
  assert(type(str) == "string")
  local t = autodiff.symbol(str, STRING)
  t.value = str
  t.eval  = function(self) return self.value end
  t.diff  = function(self, seed, result) return result end
  return t
end

-- TABLE OPERATIONS

autodiff.op[STRING] = {}
