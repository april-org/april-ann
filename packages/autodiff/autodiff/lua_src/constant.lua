local CONSTANT = autodiff.dtypes.CONSTANT
local SCALAR   = autodiff.dtypes.SCALAR
local MATRIX   = autodiff.dtypes.MATRIX
local TABLE    = autodiff.dtypes.TABLE
local STRING   = autodiff.dtypes.STRING
--
local gen_var_name = autodiff.gen_var_name
local coercion     = autodiff.coercion
local gen_op       = autodiff.gen_op

-- CONSTANT

-- declaration of a constant symbol
autodiff[CONSTANT] = function(...)
  local arg = table.pack(...)
  local result = {}
  for _,value in ipairs(arg) do
    local s = autodiff.symbol(tostring(value), CONSTANT)
    s.value = value
    s.eval  = function(self) return self.value end
    s.diff  = function(self, seed, result)
      local result = result or {}
      result[self.name] = autodiff[CONSTANT]( 0 )
      return result
    end
    s.compile = function(self,dest)
      if not self.var_name then
	self.var_name = gen_var_name()
      end
      dest:write_initial_constant(self.var_name,self:eval())
    end
    s.to_dot_string = function(self,id,parent,names,edges,idx)
      local idx = idx or 0
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
	local edge_str = string.format('%s -> %s [headlabel="%d",labeldistance=3];',
				       name_str, parent, idx)
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
  add = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT]( a:eval() + b:eval() ) end,
  sub = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT]( a:eval() - b:eval() ) end,
  pow = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT]( a:eval() ^ b:eval() ) end,
  unm = function(a)   local a=coercion(a) return autodiff[CONSTANT]( - a:eval() )     end,
  mul = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT]( a:eval() * b:eval() ) end,
  div = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT]( a:eval() / b:eval() ) end,

  -- extended math expressions
  abs = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.abs( a:eval() )) end,
  log = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.log( a:eval() ) ) end,
  exp = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.exp( a:eval() ) ) end,
  --
  sin = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.sin( a:eval() ) ) end,
  cos = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.cos( a:eval() ) ) end,
  tan = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.tan( a:eval() ) ) end,
  --
  sinh = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.sinh( a:eval() ) ) end,
  cosh = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.cosh( a:eval() ) ) end,
  tanh = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.tanh( a:eval() ) ) end,
  --
  asin = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.asin( a:eval() ) ) end,
  acos = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.acos( a:eval() ) ) end,
  atan = function(a) local a=coercion(a) return autodiff[CONSTANT]( math.atan( a:eval() ) ) end,
  -- matrix expressions
  transpose = function(a) local a=coercion(a) return autodiff[CONSTANT]( a:eval() ) end,
  fill = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT](b:eval()) end,
  cmul = function(a,b) local a,b=coercion(a),coercion(b) return autodiff[CONSTANT](a:eval()*b:eval()) end,
}
