-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixBool)

matrixBool.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrixBool.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixBool)

matrixBool.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixBool",
                                            function(value)
                                              return string.format("%d", value and 1 or 0)
  end)

matrixBool.join =
  matrix.__generic__.__make_generic_join__(matrixBool)

class.extend(matrixBool, "to_index",
             function(self)
               local self = self:squeeze()
               assert(#self:dim() == 1, "Needs a rank 1 matrix")
               local ones = self:count_ones()
               if ones == 0 then return nil end
               local result = matrixInt32(ones)
               local idx,pos=1,1
               self:map(function(x)
                   if x == 1 then result:set(pos,idx) pos=pos+1 end
                   idx=idx+1
               end)
               return result
end)

class.extend(matrixBool, "land",
             function(self,other)
               assert(class.is_a(other,matrixBool),
                      "Needs a matrixBool as argument")
               self:map(other,function(x,y)
                          return x and y
               end)
end)

class.extend(matrixBool, "lor",
             function(self,other)
               assert(class.is_a(other,matrixBool),
                      "Needs a matrixBool as argument")
               self:map(other,function(x,y)
                          return x or y
               end)
end)

matrixBool.meta_instance.__unm = function(op)
  return op:clone():complement()
end

matrixBool.meta_instance.__add = function(op1, op2)
  assert(class.is_a(op1, matrixBool), "Needs two matrixBool arguments")
  assert(class.is_a(op2, matrixBool), "Needs two matrixBool arguments")
  return op1:clone():lor(op2)
end

matrixBool.meta_instance.__mul = function(op1, op2)
  assert(class.is_a(op1, matrixBool), "Needs two matrixBool arguments")
  assert(class.is_a(op2, matrixBool), "Needs two matrixBool arguments")
  return op1:clone():land(op2)
end
