class.extend(matrixBool, "t", function(self) return self:transpose() end)

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixBool, "ascii")

matrixBool.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrixBool.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixBool)

matrixBool.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixBool",
                                            function(value)
                                              return string.format("%s", value and "T" or "F")
  end)

matrixBool.join =
  matrix.__generic__.__make_generic_join__(matrixBool)

class.extend(matrixBool, "to_index",
             april_doc{
               class = "method",
               summary = "Returns a matrixInt32 with the list of true indices",
               description = "Requires a rank 1 matrix",
               outputs = { "A matrixInt32 instance", },
             } ..
               function(self)
                 local self = self:squeeze()
                 assert(#self:dim() == 1, "Needs a rank 1 matrix")
                 local ones = self:count_ones()
                 if ones == 0 then return nil end
                 local result = matrixInt32(ones)
                 local idx,pos=0,0
                 self:map(function(x)
                     idx=idx+1
                     if x then pos=pos+1 result:set(pos,idx) end
                 end)
                 assert(pos == ones)
                 return result
end)

class.extend(matrixBool, "land",
             april_doc{
               class = "method",
               summary = "Computes component-wise AND operation IN-PLACE",
               params  = {"Another matrixBool instance",},
               outputs = {"The caller matrix",}
             } ..
               function(self,other)
                 assert(class.is_a(other,matrixBool),
                        "Needs a matrixBool as argument")
                 self:map(other,function(x,y)
                            return x and y
                 end)
                 return self
end)

class.extend(matrixBool, "lor",
             april_doc{
               class = "method",
               summary = "Computes component-wise OR operation IN-PLACE",
               params  = {"Another matrixBool instance",},
               outputs = {"The caller matrix",}
             } ..
               function(self,other)
                 assert(class.is_a(other,matrixBool),
                        "Needs a matrixBool as argument")
                 self:map(other,function(x,y)
                            return x or y
                 end)
                 return self
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

matrixBool.meta_instance.__eq = function(op1, op2)
  if class.is_a(op1, matrix) then op1 = matrixBool(op1)
  elseif class.is_a(op2, matrix) then op2 = matrixBool(op2) end
  assert(class.is_a(op1, matrixBool) and class.is_a(op2, matrixBool),
         "Needs two matrixBool arguments")
  local equals = true
  op1:map(op2,function(x,y)
            equals = equals and (x == y)
  end)
  return equals
end
