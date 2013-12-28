local CONSTANT = autodiff.dtypes.CONSTANT
local SCALAR   = autodiff.dtypes.SCALAR
local MATRIX   = autodiff.dtypes.MATRIX
local TABLE    = autodiff.dtypes.TABLE
local STRING   = autodiff.dtypes.STRING
--
local gen_var_name = autodiff.gen_var_name
local coercion     = autodiff.coercion
local gen_op       = autodiff.gen_op

-- STRING

-- the string data type exists to allow passing Lua strings to the operators

autodiff[STRING] = function(str)
  assert(type(str) == "string")
  local t = autodiff.symbol(str, STRING)
  t.value = str
  t.eval  = function(self) return self.value end
  t.diff  = function(self, seed, result) return result end
  t.compile = function(self,dest)
    if not self.var_name then
      self.var_name = gen_var_name()
    end
    dest:write_initial_constant(self.var_name,self:eval())
  end
  return t
end

-- TABLE OPERATIONS

autodiff.op[STRING] = {}
