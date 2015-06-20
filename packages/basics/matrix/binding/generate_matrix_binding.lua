function dirname(path_with_filename, sep)
  local sep=sep or'/'
  return path_with_filename:match("(.*"..sep..")") or "./"
end
local root = dirname(arg[0])

local methods = {
  float = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "isfinite", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "uniformf", "linspace", "logspace",
    "linear", "sliding_window", "is_contiguous",
    "prune_subnormal_and_check_normal", "adjust_range", "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp", "add", "scalar_add",
    "sub", "mul", "cmul", "plogp", "log", "log1p", "exp", "sqrt", "pow", "tan",
    "tanh", "atan", "atanh", "sin", "sinh", "asin", "asinh", "cos", "cosh",
    "acos", "acosh", "abs", "complement", "sign", "sum", "copy", "axpy",
    "gemm", "sparse_mm", "gemv", "ger", "dot", "scal",
    "masked_fill", "masked_copy", "div", "norm2", "inv", "logdet", "det",
    "cholesky", "svd", "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to",
  },
  double = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "isfinite", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "uniformf", "linspace", "logspace",
    "linear", "sliding_window", "is_contiguous",
    "prune_subnormal_and_check_normal", "adjust_range", "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp", "add", "scalar_add",
    "sub", "mul", "cmul", "plogp", "log", "log1p", "exp", "sqrt", "pow", "tan",
    "tanh", "atan", "atanh", "sin", "sinh", "asin", "asinh", "cos", "cosh",
    "acos", "acosh", "abs", "complement", "sign", "sum", "copy", "axpy",
    "gemm", "gemv", "ger", "dot", "scal",
    "masked_fill", "masked_copy", "div", "norm2",
    "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to",
  },
  int32_t = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "linspace",
    "linear", "sliding_window", "is_contiguous",
    "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp",
    "masked_fill", "masked_copy",
    "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to",
  },
  ComplexF = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "isfinite", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "uniformf", "linspace", "logspace",
    "linear", "sliding_window", "is_contiguous",
    "prune_subnormal_and_check_normal", "adjust_range", "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp", "add", "scalar_add",
    "sub", "mul", "cmul", "plogp", "log", "log1p", "exp", "sqrt", "pow", "tan",
    "tanh", "atan", "atanh", "sin", "sinh", "asin", "asinh", "cos", "cosh",
    "acos", "acosh", "abs", "complement", "sign", "sum", "copy", "axpy",
    "gemm", "gemv", "ger", "dot", "scal",
    "masked_fill", "masked_copy", "div", "norm2",
    "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to",
  },
  char = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding",
    "sliding_window", "is_contiguous",
    "adjust_range", "diag", "fill",
    "zeros", "ones", "equals", "clamp",
    "masked_fill", "masked_copy",
    "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to",
  },
  bool = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding",
    "sliding_window", "is_contiguous",
    "adjust_range", "diag", "fill",
    "zeros", "ones", "equals",
    "toMMap",
    "data", "convert_to",
    "to_index",
  },
}

local class_methods = {
  float = {
    "as", "deserialize", "read", "fromMMap", "fromPNM", "fromHEX",
  },
}

local class_method_binding = [[
//BIND_CLASS_METHOD %s %s
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<%s>::%s(L));
}
//BIND_END

]]

local method_binding = [[
//BIND_METHOD %s %s
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<%s>::%s(L,obj));
}
//BIND_END

]]

for _,arg in ipairs{
  { "float", "MatrixFloat", "matrix", "template.txt", "bind_matrix.lua.cc" },
  { "double", "MatrixDouble", "matrixDouble", "template.txt", "bind_matrix_double.lua.cc" },
  { "int32_t", "MatrixInt32", "matrixInt32", "template.txt", "bind_matrix_int32.lua.cc" },
  { "char", "MatrixChar", "matrixChar", "template.txt", "bind_matrix_char.lua.cc" },
  { "ComplexF", "MatrixComplex", "matrixComplex", "template.txt", "bind_matrix_complex_float.lua.cc" },
  { "bool", "MatrixBool", "matrixBool", "template.txt", "bind_matrix_bool.lua.cc" },
} do

  local T          = arg[1]
  local MATRIX_T   = arg[2]
  local MATRIX_Lua = arg[3]
  local input      = arg[4]
  local output     = arg[5]
  
  local template = io.open(root .. input):read("*a")
  local f = io.open(root .. output, "w")
  
  f:write((template:gsub("%$%$([^$]+)%$%$", { T          = T,
                                              MATRIX_T   = MATRIX_T,
                                              MATRIX_Lua = MATRIX_Lua, })))
  local generated = {}
  for _,name in ipairs(class_methods[T] or {}) do
    assert(not generated[name])
    f:write((class_method_binding:format(MATRIX_T, name, T, name)))
    generated[name] = true
  end

  f:write("///////////////////////////////////////////////////////////\n\n")
  for _,name in ipairs(methods[T] or {}) do
    assert(not generated[name])
    f:write((method_binding:format(MATRIX_T, name, T, name)))
    generated[name] = true
  end
  f:close()
end
