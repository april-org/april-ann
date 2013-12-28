local CONSTANT = autodiff.dtypes.CONSTANT
local SCALAR   = autodiff.dtypes.SCALAR
local MATRIX   = autodiff.dtypes.MATRIX
local TABLE    = autodiff.dtypes.TABLE
local STRING   = autodiff.dtypes.STRING
--
local gen_var_name = autodiff.gen_var_name
local coercion     = autodiff.coercion
local gen_op       = autodiff.gen_op

-- MATRIX

local function check_broadcast(a,b)
  local a_broadcast,b_broadcast = a.broadcast or {},b.broadcast or {}
  local ret = a
  for i=1,math.max(#a_broadcast,#b_broadcast) do
    if b_broadcast[i] then ret = b end
    if a_broadcast[i] and b_broadcast[i] then
      return false,"Only one variable could be broadcasted"
    end
  end
  return ret
end

local function check_dims(a,b)
  if a and b then
    if #a ~= #b then return false end
    for i=1,#a do
      if a[i] ~= 0 and b[i] ~= 0 then
	if a[i] ~= b[i] then return false end
      end
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
	      string.format("Incorrect dimensions, expected %s, found %s",
			    table.concat(t[i].dims or {}, "x"),
			    table.concat(m:dim(), "x")) )
      return m
    end
    local mt = getmetatable(t[i])
    mt.__call = function(t, ...) return op.slice(t, ...) end
  end
  return table.unpack(t)
end

-- MATRIX OPERATIONS

local function gen_broadcasted_symbol(name,dtype,args,
				      eval_func,compile_func)
  assert(#args==2, "Only two arguments are possible in broadcasted operations")
  local aux = assert(check_broadcast(args[1],args[2]))
  s = gen_op(name, dtype, args,
	     function(self, ...)
	       local who = aux
	       local a = self.args[1]:eval(...)
	       local b = self.args[2]:eval(...)
	       -- simplifications
	       if a == 0 then return b
	       elseif b == 0 then return a
	       end
	       --
	       local other,who_broadcast
	       if who == self.args[1] then
		 who           = a
		 other         = b
		 who_broadcast = self.args[1].broadcast or {}
	       else
		 who           = b
		 other         = a
		 who_broadcast = self.args[2].broadcast or {}
	       end
	       local who_dims      = who:dim()
	       local other_dims    = other:dim()
	       for i=1,#who_dims do
		 assert(not who_broadcast[i] or who_dims[i]==1,
			string.format("Broadcasted dimensions must be of size=1, found dims[%d]=%d",
				      i, who_dims[i]))
		 assert(who_broadcast[i] or who_dims[i]==other_dims[i],
			string.format("Not broadcasted dimensions must be equal, found broadcasted_dims[%d]=%d and dims[%d]=%d",
				      i, who_dims[i], i, other_dims[i]))
	       end
	       local result = matrix.as(other)
	       local r_sw   = result:sliding_window{ size=who_dims,
						     step=who_dims }
	       local sw     = other:sliding_window{ size=who_dims,
						    step=who_dims }
	       local r_slice,slice
	       while not r_sw:is_end() do
		 r_slice = r_sw:get_matrix(r_slice)
		 slice = sw:get_matrix(slice)
		 r_slice:copy( eval_func(slice,who) )
		 r_sw:next()
		 sw:next()
	       end
	       return result
	     end,
	     function(self, seed, result)
	       local who = aux
	       local a,b = self.args[1],self.args[2]
	       local which = 0
	       for i=1,#who.broadcast do
		 if who.broadcast[i] then
		   assert(which==0, "Only one dimension could be broadcasted")
		   which = i
		 end
	       end
	       if who == a then
		 a:diff(autodiff.op.sum(seed, which), result)
		 b:diff(seed, result)
	       else
		 a:diff(seed, result)
		 b:diff(autodiff.op.sum(seed, which), result)
	       end
	       return result
	     end,
	     function(self, dest)
	       local who = aux
	       local a,b = self.args[1],self.args[2]
	       local other,who_broadcast
	       if who == self.args[1] then
		 who           = a
		 other         = b
		 who_broadcast = self.args[1].broadcast or {}
	       else
		 who           = b
		 other         = a
		 who_broadcast = self.args[2].broadcast or {}
	       end
	       local who_dims   = gen_var_name()
	       local other_dims = gen_var_name()
	       dest:write_expr_assign(who_dims,
				      string.format("%s:dim()", who.var_name))
	       dest:write_expr_assign(other_dims,
				      string.format("%s:dim()", other.var_name))
	       dest:write_expr_assign(self.var_name,
				      string.format("matrix.as(%s)",
						    other.var_name))
	       local r_sw = gen_var_name()
	       local sw   = gen_var_name()
	       dest:write_expr_assign(r_sw,
				      string.format("%s:sliding_window{ size=%s, step=%s}",
						    self.var_name,
						    who_dims, who_dims))
	       dest:write_expr_assign(sw,
				      string.format("%s:sliding_window{ size=%s, step=%s}",
						    other.var_name,
						    who_dims, who_dims))
	       local r_slice = gen_var_name()
	       local slice   = gen_var_name()
	       dest:write_var(r_slice)
	       dest:write_var(slice)
	       dest:write_expr_block(string.format([[
while not %s:is_end() do
  %s = %s:get_matrix(%s)
  %s = %s:get_matrix(%s)
  %s:copy( %s )
  %s:next()
  %s:next()
end]],
						   r_sw,
						   r_slice, r_sw, r_slice,
						   slice, sw, slice,
						   r_slice,
						   compile_func(slice, who.var_name),
						   r_sw,
						   sw))
	     end)
  return s
end

-------------------------------------------------------------------------------

autodiff.op[MATRIX] = {
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifcations
    if b<a then a,b=b,a end
    if a == autodiff[CONSTANT](0) then return b
    elseif b == autodiff[CONSTANT](0) then return a
    end
    --
    local s
    if a.broadcast or b.broadcast then
      -- broadcasted version
      s = gen_broadcasted_symbol('+', MATRIX, {a,b},
				 function(a,b) return a+b end,
				 function(a_var_name,b_var_name)
				   return string.format("%s + %s",
							a_var_name,
							b_var_name)
				 end)
    else
      -- not broadcasted version
      s = gen_op('+', MATRIX, {a,b},
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
    end
    -- TODO: implement correctly taking into account broadcast property
    --
    -- if a.dims or b.dims then
    --   assert( check_dims(a.dims, b.dims, a.broadcast, b.broadcast),
    -- 	      "Incorrect dimensions")
    --   s:set_dims(a.dims or b.dims)
    -- end
    return s
  end,
  
  sub = function(a,b)
    local a,b = coercion(a),coercion(b)
    return a + (-1 * b)
  end,

  mul = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifcations
    if a == autodiff[CONSTANT](0) or b == autodiff[CONSTANT](0) then
      return autodiff[CONSTANT](0)
    elseif a == autodiff[CONSTANT](1) then return b
    elseif b == autodiff[CONSTANT](1) then return a
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
    if a == autodiff[CONSTANT](0) then return autodiff[CONSTANT](0)
    elseif a == autodiff[CONSTANT](1) or b == autodiff[CONSTANT](0) then
      return autodiff[CONSTANT](1)
    elseif b == autodiff[CONSTANT](1) then return a
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
		       local a    = self.args[1]
		       local seed = autodiff.op.cmul(autodiff.op.pow(a, -1), seed)
		       a:diff(seed, result)
		       return result
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
		       local a = self.args[1]
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
    -- simplification
    if a.isop and a.isop=='fill' then
      if     a.args[2] == autodiff[CONSTANT]( 1) then return  b
      elseif a.args[2] == autodiff[CONSTANT](-1) then return -b
      elseif a.args[2] == autodiff[CONSTANT]( 0) then return  a
      end
    end
    --
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
		       a:diff(autodiff.op.cmul(b,seed), result)
		       b:diff(autodiff.op.cmul(a,seed), result)
		       return result
		     end,
		     function(self, dest)
		       local a,b = self.args[1],self.args[2]
		       local str_tbl
		       if a.dtype == MATRIX and b.dtype == MATRIX then
			 str_tbl = { a.var_name, ':cmul(', b.var_name, ')' }
		       elseif (a.dtype == SCALAR or b.dtype == SCALAR) or
		       (a.dtype == CONSTANT or b.dtype == CONSTANT) then
			 str_tbl = { a.var_name, ' * ', b.var_name }
		       else
			 error(string.format("Incorrect arguments dtype: %s %s",
					     a.dtype, b.dtype))
		       end
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
    local s
    if b.dtype == SCALAR or b.dtype == CONSTANT then
      s = gen_op('fill', MATRIX, {a,b},
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
    elseif b.dtype == MATRIX then
      s = gen_op('fill', MATRIX, {a,b},
		 function(self, ...)
		   local a   = self.args[1]:eval(...)
		   local b   = self.args[2]:eval(...)
		   assert(type(a) == "matrix")
		   assert(type(b) == "matrix")
		   local result = matrix.as(a)
		   local sw = result:sliding_window{ size=b:dim(), step=b:dim() }
		   local slice
		   while not sw:is_end() do
		     slice = sw:get_matrix(slice)
		     slice:copy(b)
		     sw:next()
		   end
		   return result
		 end,
		 function(self, seed, result)
		   return result
		 end,
		 function(self, dest)
		   local a,b = self.args[1],self.args[2]
		   dest:write_expr_assign(self.var_name,
					  string.format("matrix.as( %s )",
							a.var_name))
		   local sw = gen_var_name()
		   dest:write_expr_assign(sw,
					  string.format("%s:sliding_window{ size=%s:dim(), step=%s:dim() }",
							self.var_name,
							b.var_name, b.var_name))
		   local slice = gen_var_name()
		   dest:write_var(slice)
		   dest:write_expr_block(string.format([[
while not %s:is_end() do
  %s = %s:get_matrix( %s )
  %s:copy( %s )
  %s:next()
end
]],
						       sw,
						       slice, sw, slice,
						       slice, b.var_name,
						       sw))
		 end)
    else
      error("Not recognized dtype: " .. tostring(b.dtype))
    end
    if a.dims then s:set_dims(a.dims) end
    return s
  end,
  
  sum = function(a,dim)
    local a,dim = coercion(a),dim and coercion(dim)
    local s
    if not dim then
      s = gen_op('sum', SCALAR, {a},
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
    else
      s = gen_op('sum', MATRIX, {a,dim},
		 function(self, ...)
		   local a   = self.args[1]:eval(...)
		   local dim = self.args[2]:eval(...)
		   assert(type(a)   == "matrix")
		   assert(type(dim) == "number")
		   return a:sum(dim)
		 end,
		 function(self, seed, result)
		   local a,dim = self.args[1],self.args[2]
		   a:diff(autodiff.op.fill(a, seed), result)
		   return result
		 end,
		 function(self, dest)
		   local a,dim = self.args[1],self.args[2]
		   local str_tbl = { a.var_name, ':sum(', dim.var_name, ')' }
		   dest:write_expr_assign(self.var_name,
					  table.concat(str_tbl, ""))
		 end)
    end
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
		       local aux = autodiff.op.fill(a, 0)
		       aux = autodiff.op.copy(aux,seed,table.unpack(arg))
		       a:diff(aux, result)
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
  
  sign = function(a)
    local a = coercion(a)
    local s = gen_op('sign', MATRIX, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return a:clone():sign()
		     end,
		     function(self, seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():sign()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
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
		       local sign = autodiff.op.sign(self)
		       a:diff(autodiff.op.cmul(sign,seed), result)
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

  size = function(a)
    local a = coercion(a)
    local s = gen_op('size', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return a:size()
		     end,
		     function(self, seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':size()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  mean = function(a)
    return autodiff.op.sum(a) / autodiff.op.size(a)
  end,

  max = function(a)
    local a = coercion(a)
    local s = gen_op('max', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return a:clone():max()
		     end,
		     function(self, seed, result)
		       local a = self.args[1]
		       local argmax = autodiff.op.argmax(a)
		       local out = autodiff.op.fill(a,0)
		       local out = autodiff.op.copy(out, seed, argmax)
		       a:diff(out, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { a.var_name, ':clone():max()' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,

  argmax = function(a)
    local a = coercion(a)
    local s = gen_op('argmax', SCALAR, {a},
		     function(self, ...)
		       local a = self.args[1]:eval(...)
		       assert(type(a) == "matrix")
		       return select(2,a:clone():max())
		     end,
		     function(self, seed, result)
		       return result
		     end,
		     function(self, dest)
		       local a = self.args[1]
		       local str_tbl = { 'select(2,', a.var_name, ':clone():max())' }
		       dest:write_expr_assign(self.var_name,
					      table.concat(str_tbl, ""))
		     end)
    if a.dims then s:set_dims(a.dims) end
    return s
  end,
  
}
