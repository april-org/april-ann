local CONSTANT = autodiff.dtypes.CONSTANT
local SCALAR   = autodiff.dtypes.SCALAR
local MATRIX   = autodiff.dtypes.MATRIX
local TABLE    = autodiff.dtypes.TABLE
local STRING   = autodiff.dtypes.STRING
--
local gen_var_name = autodiff.gen_var_name
local coercion     = autodiff.coercion
local gen_op       = autodiff.gen_op

-- TABLE

-- the table data type exists to allow passing Lua tables to the operators

autodiff[TABLE] = function(tbl)
  assert(type(tbl) == "table")
  local t = autodiff.symbol(table.tostring(tbl), TABLE)
  t.value = tbl
  t.eval  = function(self) return self.value end
  t.diff  = function(self, seed, result) return result end
  t.compile = function(self,dest)
    if not self.var_name then
      self.var_name = gen_var_name()
    end
    dest:write_initial_constant(self.var_name,
				string.format("%s",
					      table.tostring(self:eval())))
  end
  return t
end

-- TABLE OPERATIONS

autodiff.op[TABLE] = {}
