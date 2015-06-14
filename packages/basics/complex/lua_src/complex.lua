-- OVERWRITTING TOSTRING FUNCTION
local function get_operands(op1,op2)
  if not class.is_a(op1,complex) then op1,op2=op2,op1 end
  local t2 = type(op2)
  if t2 == "number" then op2 = complex(op2, 0.0)
  elseif t2 == "string" then op2 = complex(op2) end  
  return op1,op2
end

class.extend(complex, "ctor_name", function(self) return "complex" end)
class.extend(complex, "ctor_params", function(self) return self:plane() end)
class.extend(complex, "to_lua_string",
             function(self) return "complex(%g,%g)"%{ self:plane() } end)

complex.meta_instance.__tostring = function(self)
  return string.format("% g%+gi", self:plane())
end

complex.meta_instance.__eq = function(op1, op2)
  local op1,op2 = get_operands(op1,op2)
  return op1:eq(op2)
end

complex.meta_instance.__add = function(op1, op2)
  local op1,op2 = get_operands(op1,op2)
  return op1:add(op2)
end

complex.meta_instance.__sub = function(op1, op2)
  local op1,op2 = get_operands(op1,op2)
  return op1:sub(op2)
end

complex.meta_instance.__mul = function(op1, op2)
  local op1,op2 = get_operands(op1,op2)
  return op1:mul(op2)
end

complex.meta_instance.__div = function(op1, op2)
  local op1,op2 = get_operands(op1,op2)
  return op1:div(op2)
end

complex.meta_instance.__unm = function(op)
  return op:neg()
end
