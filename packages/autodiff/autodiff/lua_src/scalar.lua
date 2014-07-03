local CONSTANT = autodiff.dtypes.CONSTANT
local SCALAR   = autodiff.dtypes.SCALAR
local MATRIX   = autodiff.dtypes.MATRIX
local TABLE    = autodiff.dtypes.TABLE
local STRING   = autodiff.dtypes.STRING
--
local gen_var_name = autodiff.gen_var_name
local coercion     = autodiff.coercion
local gen_op       = autodiff.gen_op

-- SCALAR

autodiff[SCALAR] = function(names) return autodiff.symbol(names, SCALAR) end

-- SCALAR OPERATIONS

autodiff.op[SCALAR] = {
  
  -- basic math operations
  
  add = function(a,b)
    local a,b = coercion(a),coercion(b)
    -- simplifications
    if b<a then a,b=b,a end
    if a == autodiff[CONSTANT](0) then return b
    elseif b == autodiff[CONSTANT](0) then return a
    elseif b == -a or -b == a then return autodiff[CONSTANT](0)
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
    if a == autodiff[CONSTANT](0) or b == autodiff[CONSTANT](0) then
      return autodiff[CONSTANT](0)
    elseif a == autodiff[CONSTANT](1) then return b
    elseif b == autodiff[CONSTANT](1) then return a
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
    if a == autodiff[CONSTANT](0) then
      return autodiff[CONSTANT](0)
    elseif a == autodiff[CONSTANT](1) then return autodiff[CONSTANT](1)
    elseif b == autodiff[CONSTANT](0) then return autodiff[CONSTANT](1)
    elseif b == autodiff[CONSTANT](1) then return a
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
    if a == autodiff[CONSTANT](0) then return autodiff[CONSTANT](1) end
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
		       a:diff(a*seed/self, result)
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

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function mul_scalar_optimization(...)
  for node in autodiff.graph_iterators.post_order_traversal(...) do
    if node.isop == '*' and node.dtype == SCALAR then
      local constant = autodiff[CONSTANT](1)
      local exp = autodiff[CONSTANT](0)
      local vd = {}
      local function count(a,b)
	local b = b or autodiff[CONSTANT](1)
	if a.dtype == CONSTANT then constant = constant * a^b
	else vd[a] = (vd[a] or autodiff[CONSTANT](0)) + b end
      end
      local function child_traverse(child)
	if child.isop == '*' then
	  for i,v in child:arg_ipairs() do child_traverse(v) end
	elseif child.isop == '^' then
	  count(child.args[1], child.args[2])
	elseif child.isop == 'exp' then
	  exp = exp + child.args[1]
	else count(child)
	end
      end
      child_traverse(node)
      -- modify the current symbol with all the stored multiplications
      local vars = iterator(pairs(vd)):select(1):table()
      -- canonical form (sorted)
      table.sort(vars)
      -- new symbol
      local new_node = constant * autodiff.op.exp(exp)
      for i,v in ipairs(vars) do new_node = new_node * (v^vd[v]) end
      -- substitution
      if new_node ~= node then node:replace(new_node) end
    end -- if node.isop == '*'
  end -- for node in post_order_traversal
end

local function pow_scalar_optimization(...)
  for node in autodiff.graph_iterators.post_order_traversal(...) do
    if node.isop == '^' and node.dtype == SCALAR then
      local a,b = node.args[1],node.args[2]
      if a.isop == '^' and a.dtype == SCALAR then
	local new_node = a.args[1]^(b*a.args[2])
	-- substitution
	if new_node ~= node then node:replace(new_node) end
      end
    end -- if node.isop == '^'
  end -- for node in post_order_traversal
end

local function exp_scalar_optimization(...)
  for node in autodiff.graph_iterators.post_order_traversal(...) do
    if node.isop == '^' and node.dtype == SCALAR then
      local a,b = node.args[1],node.args[2]
      if a.isop == 'exp' and a.dtype == SCALAR then
	local new_node = autodiff.op.exp(a.args[1]*b)
	-- substitution
	if new_node ~= node then node:replace(new_node) end
      end
    end -- if node.isop == '^'
  end -- for node in post_order_traversal
end

-- register optimizations
autodiff.optdb.register_global(mul_scalar_optimization)
autodiff.optdb.register_global(pow_scalar_optimization)
autodiff.optdb.register_global(exp_scalar_optimization)
