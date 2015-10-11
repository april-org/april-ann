-- GENERIC __CALL METAMETHOD
matrix.__generic__ = matrix.__generic__ or {}

matrix.__generic__.__make_generic_call__ = function(matrix_class)
  assert(matrix_class and class.is_class(matrix_class),
         "__make_generic_call__: Needs a class table as argument")
  class.extend_metamethod(matrix_class, "__call",
                          matrix_class.__call_function__)
end

matrix.__generic__.__make_generic_index__ = function(matrix_class)
  assert(matrix_class and class.is_class(matrix_class),
         "__make_generic_index__: Needs a class table as argument")
  class.declare_functional_index(matrix_class,
                                 matrix_class.__index_function__)
end

matrix.__generic__.__make_generic_newindex__ = function(matrix_class)
  assert(matrix_class and class.is_class(matrix_class),
         "__make_generic_newindex__: Needs a class table as argument")
  return matrix_class.__newindex_function__
end
