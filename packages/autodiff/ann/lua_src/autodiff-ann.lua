-- This file implements specific operations related with ANNs. Only the most
-- numerically inestable are implemented.

local AD = autodiff
AD.ann   = AD.ann or {}

-- LOG SOFTMAX ACTIVATION FUNCTION WITH CROSS-ENTROPY LOSS
function AD.ann.cross_entropy_log_softmax(target,net)
  local t = AD.coercion(target)
  local n = AD.coercion(net)
  local s = AD.gen_op('CE_log_softmax', AD.dtypes.SCALAR, {t,n},
		      function(self, ...)
			local t   = self.args[1]:eval(...)
			local n   = self.args[2]:eval(...)
			local max = n:max()
			local out = n - max
			local sum = out:clone():exp():sum()
			local out = out - math.log(sum)
			local L   = -out:cmul(t):sum()
			return L
		      end,
		      function(self, seed, result)
			local t = self.args[1]
			local n = self.args[2]
			local dself = AD.op.exp(AD.ann.log_softmax(n)) - t
			n:diff(AD.op.cmul(seed, dself), result)
			return result
		      end,
		      function(self, dest)
			local t   = self.args[1]
			local n   = self.args[2]
			local max = AD.gen_var_name()
			local sum = AD.gen_var_name()
			dest:write_expr_assign(max,
					       string.format("%s:max()",
							     n.var_name))
			dest:write_expr_assign(self.var_name,
					       string.format("(%s - %s)",
							     n.var_name, max))
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
