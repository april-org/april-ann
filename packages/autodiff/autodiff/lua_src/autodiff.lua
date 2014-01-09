autodiff                 = autodiff or {}
autodiff.op              = autodiff.op or {}
autodiff.graph_iterators = autodiff.graph_iterators or {}
autodiff.optdb           = autodiff.optdb or {}

local global_optimizations_db = {}
local local_optimizations_db  = {}

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- auxiliary functions for variable name generation
local gen_var_name
local reset_var_id
do
  local var_name_id=0
  reset_var_id = function() var_name_id = 0 end
  gen_var_name = function()
    var_name_id=var_name_id+1
    return string.format("__var%d__", var_name_id)
  end
end

autodiff.gen_var_name = gen_var_name

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
		 local obj = { f=f, indent=1, active_vars={}, cache_counts={},
			       declared_expressions = {} }
		 setmetatable(obj,compiler_out_mt)
		 obj.f:write("return function(arg,cache)\n")
		 return obj
	       end,
	     })

-- methods
compiler_out_mt.__index = {
  -- basic methods
  declared_expression = function(self,var_name)
    return self.declared_expressions[var_name]
  end,
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
    self.declared_expressions = {}
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
    if self.cache_counts[var_name] > (parent_count or 1) then
      self:write_indent()
      self.f:write(string.format("if not cache[%q] then\n", var_name))
      self.indent = self.indent + 1
    end
  end,
  write_expr_line = function(self, expression)
    self:write_indent()
    self.f:write(string.format("%s\n", expression))
  end,
  write_expr_block = function(self, block)
    for line in block:lines_of() do
      self:write_indent()
      self.f:write(string.format("%s\n", line))
    end
  end,
  write_expr_assign = function(self, var_name, expression)
    self:write_indent()
    if not self.active_vars[var_name] then
      self.f:write("local ")
    end
    self.f:write(string.format("%s = (%s)\n", var_name, expression))
    self.active_vars[var_name] = true
  end,
  end_expression = function(self, var_name, parent_count, childs)
    assert(self.active_vars[var_name],
	   "Declare expression vars before writing them")
    -- The same parent and children cache count means that they are dependent,
    -- so the children always come with the same parent. If the counts are
    -- different, then the children has a dependence in paths different than the
    -- current parent. In the first case, it is not necessary to introduce a new
    -- block, because children and parent always come together. In the second
    -- case, a block with a cache check is needed.
    if self.cache_counts[var_name] > (parent_count or 1) then
      self:write_indent()
      self.f:write(string.format("cache[%q] = %s\n", var_name, var_name))
      self.indent = self.indent - 1
      self:write_indent()
      self.f:write(string.format("else -- if not cache[%q]\n", var_name))
      self.indent = self.indent + 1
      if childs then
	local function get_child_from_cache(v)
	  if v.isop then
	    self:write_indent()
	    self.f:write(string.format("%s = cache[%q]\n",
				       v.var_name, v.var_name))
	    self.declared_expressions[v.var_name] = true
	    for i=1,#v.args do get_child_from_cache(v.args[i]) end
	  end
	end
	for i=1,#childs do get_child_from_cache(childs[i]) end
      end
      self:write_indent()
      self.f:write(string.format("%s = cache[%q]\n", var_name, var_name))
      self.indent = self.indent - 1
      self:write_indent()
      self.f:write(string.format("end -- if not cache[%q] else ... end\n",
	  var_name))
    end
    self.declared_expressions[var_name] = true
  end,
}
------------------------------------------------------------------------------

-- dtype constants
local CONSTANT = 'constant'
local SCALAR   = 'scalar'
local MATRIX   = 'matrix'
local TABLE    = 'table'
local STRING   = 'string'

autodiff.dtypes = {
  CONSTANT = CONSTANT,
  SCALAR   = SCALAR,
  MATRIX   = MATRIX,
  TABLE    = TABLE,
  STRING   = STRING,
}

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- auxiliar function which inserts gradient of a given symbol name, accumulating
-- gradients which come from different graph paths
local function insert_grad(t, key, value)
  assert(value, "Find nil value while processing gradient of " .. tostring(key))
  local t = t or {}
  t[key] = (not t[key] and value) or (t[key] + value)
  return t
end

-- all the symbols declared in a program will be stored here, so, symbol names
-- couldn't be reused
local SYMBOLS = {}
setmetatable(SYMBOLS, { __mode="v" })

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
  assert(dtype, string.format("Found nil dtype at argument: %s\n", arg[1]))
  for i=2,#arg do
    local argi_dtype = (type(arg[i]) == "number" and CONSTANT) or arg[i].dtype
    assert(argi_dtype,
	   string.format("Found nil dtype at argument: %s\n", arg[i]))
    dtype = infer_table[dtype][argi_dtype]
  end
  return dtype
end

-- a function which declares symbols given its name and its dtype. A symbol
-- has two basic methods:
--
-- e:eval(values,cache) => evaluates the expression using the given variable
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
  if SYMBOLS[name] and SYMBOLS[name].name == name then
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
      name      = name,   -- the name identifies the symbol
      dtype     = dtype,  -- indicates the expected type of the symbol
      issymbol  = true,   -- indicates that this table is a symbol
      dims      = nil,    -- it is possible to write the dimensions if needed the
      broadcast = nil,    -- explicit indicate of broadcasting
      --
      arg_ipairs = function() return ipairs{} end,
      --
      replace = function(self,new_self)
	collectgarbage("collect")
	local new_self = autodiff.coercion(new_self)
	if self ~= new_self then
	  for name,v in pairs(SYMBOLS) do
	    for i,child in ipairs(v.args or {}) do
	      if child == self then v.args[i] = new_self end
	    end
	    v:unmark()
	  end
	  for name,v in pairs(SYMBOLS) do
	    v:generate_name()
	    SYMBOLS[v.name] = v
	  end
	end
	collectgarbage("collect")
	return self
      end,
      --
      unmark = function(self) self.visited = nil end,
      generate_name = function() end,
      -- following method removes the var_name associated with the compilation
      -- of the symbol
      clear_var_name = function(self) self.var_name = nil return self end,
      -- modifies the dimensions of the symbol shape
      set_dims = function(self,...)
	self.dims = table.pack(...)
	if type(self.dims[1]) == "table" then
	  assert(#self.dims == 1,
		 "set_dims accepts ONE table or a MULTIPLE numbers list")
	  self.dims = self.dims[1]
	end
	return self
      end,
      -- indicates if it is possible to broadcast the result over each dimension
      set_broadcast = function(self,...)
	self.broadcast = table.pack(...)
	if type(self.broadcast[1]) == "table" then
	  assert(#self.broadcast == 1,
		 "set_broadcast accepts ONE table or a MULTIPLE numbers list")
	  self.broadcast = self.broadcast[1]
	end
	return self
      end,
      -- ignore the gradient computation, take the seed as gradient
      ignore_gradient = function(self)
	self.diff = function(self,seed,result)
	  local seed = seed or autodiff.op.fill(self,1)
	  insert_grad(result, self.name, seed)
	  for _,v in ipairs(self.args or {}) do v:diff(seed,result) end
	  return result
	end
      end,
      -- default eval function, returns the value stored at values table
      eval = function(self,values)
	local m = values[self.name] or error("Undefined value " .. self.name)
	self.last = m
	return m
      end,
      -- default diff function, introduces the given seed at the result table
      diff = function(self, seed, result)
	local seed = seed or autodiff.op.fill(self,1)
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
      to_dot_string = function(self,id,parent,names,edges,idx)
	local idx = idx or 0
	local aux = { string.format("%s [shape=box];", name) }
	if parent then
	  local edge_str = string.format('%s -> %s [headlabel="%d",labeldistance=3];',
					 name, parent, idx)
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
  if type(a) == "number" then return autodiff[CONSTANT](a)
  elseif type(a) == "table" and not a.issymbol then return autodiff[TABLE](a)
  elseif type(a) == "string" then return autodiff[STRING](a)
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
    if type(name) == "table" then name = name.name end
    assert(type(name)=="string", "Needs a symbol table or symbol name string")
    SYMBOLS[name] = nil
  end
end

-- returns a symbol given its name
function autodiff.get(name)
  return SYMBOLS[name]
end

autodiff.coercion = coercion

-- this functions returns a new operation with the given data
function autodiff.gen_op(name, dtype, args,
			 eval_func, diff_func, compile)
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
  --
  s.unmark = function(self)
    if self.visited then
      self.visited = nil
      iterator(ipairs(self.args)):select(2):call('unmark'):apply()
    end
  end
  --
  s.generate_name = function(self)
    if not self.visited then
      self.visited = true
      iterator(self:arg_ipairs()):select(2):call('generate_name'):apply()
      self.name = string.format("(%s %s)", self.isop,
				iterator(self:arg_ipairs()):select(2):
				map(tostring):concat(" "))
    end
  end
  --
  s.arg_ipairs = function(self) return ipairs(self.args) end
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
    if not dest:declared_expression(self.var_name) then
      dest:begin_expression(self.var_name, parent_count)
      -- compiles the arguments list
      iterator(self:arg_ipairs()):select(2):
      call('compile',dest,dest:get_cache_count(self.var_name)):apply()
      -- compiles the operation expression itself
      compile(self, dest)
      dest:end_expression(self.var_name, parent_count, self.args)
    end
  end
  -- removes the associated var_name, and calls the clear_var_name of its
  -- arguments
  s.clear_var_name = function(self)
    self.var_name = nil
    iterator(self:arg_ipairs()):select(2):call('clear_var_name'):apply()
  end
  -- auxiliary function for debugging purposes
  s.to_dot_string = function(self,id,parent,names,edges,idx)
    local idx   = idx   or 0
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
      local edge_str = string.format('%s -> %s [headlabel="%d",labeldistance=3];',
				     name_str, parent, idx)
      if not edges[edge_str] then
	table.insert(aux, edge_str)
	edges[edge_str] = true
      end
    end
    for i,v in self:arg_ipairs() do
      local str = v:to_dot_string(id,name_str,names,edges,i)
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
function autodiff.func(s, args, shared_values, optimize)
  local optimize = (optimize==nil and true) or optimize
  assert(type(s) == "table")
  if s.issymbol then s = { s } end
  -- optimize all the given symbols
  if optimize then
    for i=1,#s do s[i] = autodiff.optimize(s[i]) end
  end
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
  local filename = os.tmpname()
  local dest = compiler_out(filename)
  -- FIRST, traverse the symbols to acquire cache counts, which will be used to
  -- optimize the produced code, and checks if all the not op symbol variables
  -- are given as argument or as shared_value (symbols_dict)
  local function count_cache(v,dest)
    if not v.var_name then v.var_name = gen_var_name() end
    dest:count_cache(v.var_name)
    if v.isop then
      for j,v2 in ipairs(v.args) do count_cache(v2,dest) end
    else assert(v.dtype==CONSTANT or v.dtype==STRING or v.dtype==TABLE or symbols_dict[v.name],
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
      local function get_vars(v,dest)
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
  -- the returned a callable table, with a closure, which inserts the input
  -- arguments and shared_values in a dictionary; this dictionary is passed to
  -- each of the previously compiled functions
  local ret = {
    program       = program,
    inputs        = args,
    outputs       = s,
    shared_values = shared_values,
    funcs         = funcs,
  }
  setmetatable(ret,{
		 __call = function(self, ...)
		   local shared_values = self.shared_values
		   local args  = self.inputs
		   local funcs = self.funcs
		   local args2 = table.pack(...)
		   local cache = {}
		   if #args2 == #args+1 then
		     cache = table.remove(args2, #args2)
		     assert(type(cache) == "table", "Expected a cache table as last argument")
		   end
		   assert(#args == #args2,
			  string.format("Incorrect number of arguments, expected %d, found %d\n",
					#args, #args2))
		   local values = {} for k,v in ipairs(args) do values[v.name] = args2[k] end
		   for k,v in pairs(shared_values) do values[k] = v end
		   local ret = {} for i,f in ipairs(funcs) do ret[i]=f(values,cache) end
		   return table.unpack(ret)
		 end,
		   })
  return ret
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
		      map(function(i,s)
			    april_assert(type(s)=="table" and s.issymbol,
					 "Found a not symbol variable at position %d",
					 i)
			    return all_diff[s.name] or
			      error("Gradient of " .. s.name .. " not implemented, or symbol not found")
			  end):
		      table())
end

-- dummy operation, adds a sentinel operation useful to perform optimizations
local function dummy(s)
  return gen_op('sentinel', s.dtype, {s})
end

-- applies optimizations
function autodiff.optimize(s)
  local memory   = {} -- used to avoid possible oscillations
  local graph_it = autodiff.graph_iterators.post_order_traversal
  repeat
    local current_name = s.name
    memory[current_name] = true
    -- sentinel dummy operation
    local dummy_s = dummy(s)
    for which,opt_func in ipairs(local_optimizations_db) do
      -- apply optimization over the sentinel argument
      for node,parent,child_idx in graph_it(dummy_s.args[1], dummy_s, 1) do
	opt_func(parent, node, child_idx)
      end
    end
    for which,opt_func in ipairs(global_optimizations_db) do
      -- apply optimization over the sentinel argument
      opt_func(dummy_s.args[1], dummy_s, 1)
    end
    -- retrieve the sentinel argument and substitutes the original operation
    s = dummy_s.args[1]
  until s.name == current_name or memory[s.name]
  return s
end

-- auxiliary function for debugging purposes
function autodiff.dot_graph(s, filename)
  local f = io.open(filename, "w")
  f:write("digraph g {\nrankdir=BT;\n")
  f:write( s:to_dot_string() )
  f:write("}\n")
  f:close()
end

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

-- auxiliary metatable for the autodiff.op table
setmetatable(autodiff.op,
	     {
	       __index = function(s,key)
		 return rawget(s,key) or
		   function(...)
		     assert(select('#',...) > 0,
			    "Incorrect number of arguments")
		     local dtype = infer(...)
		     assert(dtype, "Found nil dtype")
		     local t = assert(autodiff.op[dtype],
				      "Incorrect type " .. (dtype or "nil"))
		     local t = assert(t[key],
				      "Operation: " .. key .. " not implemented for type: " .. dtype)
		     return t(...)
		   end
	       end,
	     })

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

function autodiff.graph_iterators.pre_order_traversal(s,parent,k)
  local visited = {}
  local function yield_iterator(s,parent,i,depth)
    if not visited[s] then
      visited[s] = true
      coroutine.yield(s,parent,i,depth)
      for j,child in s:arg_ipairs() do yield_iterator(child,s,j,depth+1) end
    end
  end
  return coroutine.wrap(function() yield_iterator(s,parent,k,1) end)
end

function autodiff.graph_iterators.post_order_traversal(s,parent,k)
  local visited = {}
  local function yield_iterator(s,parent,i,depth)
    if not visited[s] then
      visited[s] = true
      for j,child in s:arg_ipairs() do yield_iterator(child,s,j,depth+1) end
      coroutine.yield(s,parent,i,depth)
    end
  end
  return coroutine.wrap(function() yield_iterator(s,parent,k,1) end)
end

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

function autodiff.optdb.register_global(opt_func)
  table.insert(global_optimizations_db, opt_func)
end

function autodiff.optdb.register_local(opt_func)
  table.insert(local_optimizations_db, opt_func)
end

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function add_general_optimization(...)
  for node in autodiff.graph_iterators.post_order_traversal(...) do
    if node.isop == '+' then
      local constant = autodiff[CONSTANT](0)
      local vd = {}
      local function count(a,b)
	local b = b or autodiff[CONSTANT](1)
	assert(b.dtype == CONSTANT) -- sanity check
	if a.dtype == CONSTANT then constant = constant + a*b
	else vd[a] = (vd[a] or autodiff[CONSTANT](0)) + b end
      end
      local function child_traverse(child)
	if child.isop == '+' then
	  for i,v in child:arg_ipairs() do child_traverse(v) end
	elseif child.isop == '*' then
	  local a,b = child.args[1],child.args[2]
	  if b.dtype == CONSTANT then count(a,b)
	  elseif a.dtype == CONSTANT then count(b,a)
	  else count(child)
	  end
	else count(child)
	end
      end
      child_traverse(node)
      -- modify the current symbol with all the stored additions
      local vars = iterator(pairs(vd)):select(1):table()
      -- canonical form (sorted)
      table.sort(vars)
      -- new symbol
      local new_node = constant
      for i,v in ipairs(vars) do new_node = new_node + (vd[v]*v) end
      -- substitution
      if new_node ~= node then node:replace(new_node) end
    end -- if node.isop == '+'
  end -- for node in post_order_traversal
end

-- optimization registration
autodiff.optdb.register_global(add_general_optimization)
