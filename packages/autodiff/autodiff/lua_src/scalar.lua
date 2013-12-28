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

