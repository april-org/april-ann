class.extend_metamethod(matrixBool, "__len", function(self) return self:dim(1) end)
class.extend_metamethod(matrixBool, "__ipairs",
                        function(self)
                          return function(self,i)
                            i = i+1
                            if i <= #self then return i,self[i] end
                          end, self, 0
end)

class.extend(matrixBool, "t", matrixBool.."transpose")

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixBool, "ascii")

matrix.__generic__.__make_index_methods__(matrixBool)

class.extend(matrixBool, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

april_set_doc(matrixBool.."to_index",
              {
                class = "method",
                summary = "Returns a matrixInt32 with the list of true indices",
                description = "Requires a rank 1 matrix",
                outputs = { "A matrixInt32 instance", },
})

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
  if type(op1) == "number" or type(op2) == "number" then return false end
  return op1:equals(op2)
end

matrixBool.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrixBool.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixBool)

matrix.__generic__.__make_generic_index__(matrixBool)

matrixBool.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixBool",
                                            function(value)
                                              return string.format("%s", value and "T" or "F")
  end)

matrixBool.join =
  matrix.__generic__.__make_generic_join__(matrixBool)
