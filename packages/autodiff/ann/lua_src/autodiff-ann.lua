-- This file implements specific operations related with ANNs. Only the most
-- numerically inestable are implemented.

local AD = autodiff
AD.ann   = AD.ann or {}

-- LOGISTIC ACTIVATION FUNCTION
function AD.ann.logistic(a)
  local a = AD.coercion(a)
  local s = AD.gen_op('logistic', AD.dtypes.MATRIX, {a},
		      function(self, ...)
			local a = self.args[1]:eval(...)
			return a:clone():scal(-1):exp():scalar_add(1.0):div(1.0)
		      end,
		      function(self, seed, result)
			local a     = self.args[1]
			local dself = AD.op.cmul(self, (1-self))
			a:diff(AD.op.cmul(seed, dself), result)
			return result
		      end,
		      function(self, dest)
			local a = self.args[1]
			local tbl = { a.var_name, ":clone()", ":scal(-1)",
				      ":exp()", ":scalar_add(1.0)",
				      ":div(1.0)" }
			dest:write_expr_assign(self.var_name,
					       table.concat(tbl, ""))
		      end)
  return s  
end

-- LOG SOFTMAX ACTIVATION FUNCTION WITH CROSS-ENTROPY LOSS
function AD.ann.cross_entropy_log_softmax(input, target)
  local i = AD.coercion(input)
  local t = AD.coercion(target)
  local s = AD.gen_op('CE_log_softmax', AD.dtypes.SCALAR, {i,t},
		      function(self, ...)
			local i   = self.args[1]:eval(...)
			local t   = self.args[2]:eval(...)
			local max = i:max()
			local out = i - max
			local sum = out:clone():exp():sum()
			local out = out - math.log(sum)
			local L   = -out:cmul(t):sum()
			return L
		      end,
		      function(self, seed, result)
			local i = self.args[1]
			local t = self.args[2]
			local dself = AD.op.exp(AD.ann.log_softmax(i)) - t
			i:diff(AD.op.cmul(seed, dself), result)
			return result
		      end,
		      function(self, dest)
			local i   = self.args[1]
			local t   = self.args[2]
			local max = AD.gen_var_name()
			local sum = AD.gen_var_name()
			dest:write_expr_assign(max,
					       string.format("%s:max()",
							     i.var_name))
			dest:write_expr_assign(self.var_name,
					       string.format("(%s - %s)",
							     i.var_name, max))
			dest:write_expr_assign(sum,
					       string.format("%s:clone():exp():sum()",
							     self.var_name))
			dest:write_expr_assign(self.var_name,
					       string.format("%s - math.log(%s)",
							     self.var_name,
							     sum))
			dest:write_expr_assign(self.var_name,
					       string.format("-%s:cmul(%s):sum()",
							     self.var_name,
							     t.var_name))
		      end)
  return s
end

-- LOG-SOFTMAX
function AD.ann.log_softmax(a)
  local a = AD.coercion(a)
  local s = AD.gen_op('log_softmax', AD.dtypes.MATRIX, {a},
		      function(self, ...)
			local a   = self.args[1]:eval(...)
			local max = a:max()
			local out = a - max
			local sum = out:clone():exp():sum()
			local out = out - math.log(sum)
			return out
		      end,
		      function(self, seed, result)
			local a = self.args[1]
			local dself = AD.op.cmul(AD.op.exp(self), (1-AD.op.exp(self)))
			a:diff(AD.op.cmul(seed, dself), result)
			return result
		      end,
		      function(self, dest)
			local a   = self.args[1]
			local max = AD.gen_var_name()
			local sum = AD.gen_var_name()
			dest:write_expr_assign(max,
					       string.format("%s:max()",
							     a.var_name))
			dest:write_expr_assign(self.var_name,
					       string.format("(%s - %s)",
							     a.var_name, max))
			dest:write_expr_assign(sum,
					       string.format("%s:clone():exp():sum()",
							     self.var_name))
			dest:write_expr_assign(self.var_name,
					       string.format("%s - math.log(%s)",
							     self.var_name,
							     sum))
		      end)
  return s
end
