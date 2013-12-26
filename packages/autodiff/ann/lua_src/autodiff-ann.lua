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
function AD.ann.cross_entropy_log_softmax(input, target, dim)
  local i   = AD.coercion(input)
  local t   = AD.coercion(target)
  local dim = AD.coercion(dim)
  local output = AD.ann.log_softmax(i,dim)
  -- ignore the gradient of softmax, it is computed at the loss function
  output:ignore_gradient()
  -- cross_entropy
  s = AD.gen_op('CE', AD.dtypes.MATRIX, {output,t,dim},
		function(self, ...)
		  local i   = self.args[1]:eval(...)
		  local t   = self.args[2]:eval(...)
		  local dim = self.args[3]:eval(...)
		  return -i:cmul(t):sum(3-dim)
		end,
		function(self, seed, result)
		  local i = self.args[1]
		  local t = self.args[2]
		  local dself = AD.op.exp(i) - t
		  seed:set_broadcast( (dim==1) and true or false,
				      (dim==2) and true or false )
		  local seed  = AD.op.fill(i, seed)
		  i:diff(AD.op.cmul(seed, dself), result)
		  return result
		end,
		function(self, dest)
			local i   = self.args[1]
			local t   = self.args[2]
			local dim = self.args[3]
			dest:write_expr_assign(self.var_name,
					       string.format("-%s:cmul(%s):sum(3 - %s)",
							     i.var_name,
							     t.var_name,
							     dim.var_name))
		      end)
  return s
end

-- LOG-SOFTMAX
function AD.ann.log_softmax(a,dim)
  local a,dim = AD.coercion(a),AD.coercion(dim)
  local s = AD.gen_op('log_softmax', AD.dtypes.MATRIX, {a,dim},
		      function(self, ...)
			local i   = self.args[1]:eval(...)
			local dim = self.args[2]:eval(...)
			local other_dim = 3 - dim
			local max = i:max(other_dim)
			local out = i:clone()
			local slice
			for k=1,out:dim(dim) do
			  slice = out:select(dim,k,slice)
			  slice:scalar_add(-max:get(
					     (dim==1 and k) or 1,
					     (dim==2 and k) or 1))
			end
			out:exp()
			for k=1,out:dim(dim) do
			  slice = out:select(dim,k,slice)
			  local sum = slice:sum()
			  slice:scalar_add(-math.log(sum))
			end
			return out
		      end,
		      function(self, seed, result)
			local a = self.args[1]
			local dself = AD.op.cmul(AD.op.exp(self), (1-AD.op.exp(self)))
			a:diff(AD.op.cmul(seed, dself), result)
			return result
		      end,
		      function(self, dest)
			local i   = self.args[1]
			local dim = self.args[2]
			local max = AD.gen_var_name()
			local sum = AD.gen_var_name()
			local other_dim = AD.gen_var_name()
			dest:write_expr_assign(other_dim,
					       string.format("3 - %s",
							     dim.var_name))
			dest:write_expr_assign(self.var_name,
					       string.format("%s:clone()",
							     i.var_name))
			local max = AD.gen_var_name()
			dest:write_expr_assign(max,
					       string.format("%s:max(%s)",
							     self.var_name,
							     other_dim))
			local slice = AD.gen_var_name()
			dest:write_var(slice)
			dest:write_expr_block(string.format([[
if %s == 1 then
  for k=1,%s:dim(%s) do
    %s = %s:select(%s,k,%s)
    %s:scalar_add(-%s:get(k,1))
  end
else
  for k=1,%s:dim(%s) do
    %s = %s:select(%s,k,%s)
    %s:scalar_add(-%s:get(1,k))
  end
end
]],
							    dim.var_name,
							    self.var_name, dim.var_name,
							    slice, self.var_name, dim.var_name, slice,
							    slice, max,
							    self.var_name, dim.var_name,
							    slice, self.var_name, dim.var_name, slice,
							    slice, max))
			local sum_exp = AD.gen_var_name()
			dest:write_expr_assign(sum_exp,
					       string.format("%s:clone():exp()",
							     self.var_name))
			dest:write_expr_assign(sum_exp,
					       string.format("%s:sum(%s)",
							     sum_exp,
							     other_dim))
			dest:write_expr_block(string.format([[
if %s == 1 then
  for k=1,%s:dim(%s) do
    %s = %s:select(%s,k,%s)
    %s:scalar_add( -math.log(%s:get(k,1)) )
  end
else
  for k=1,%s:dim(%s) do
    %s = %s:select(%s,k,%s)
    %s:scalar_add( -math.log(%s:get(1,k)) )
  end
end
]],
							    dim.var_name,
							    self.var_name, dim.var_name,
							    slice, self.var_name, dim.var_name, slice,
							    slice, sum_exp,
							    self.var_name, dim.var_name,
							    slice, self.var_name, dim.var_name, slice,
							    slice, sum_exp))
		      end)
  return s
end
