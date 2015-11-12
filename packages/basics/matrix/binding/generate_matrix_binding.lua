function dirname(path_with_filename, sep)
  local sep=sep or'/'
  return path_with_filename:match("(.*"..sep..")") or "./"
end
function basename(path)
  local name = string.match(path, "([^/]+)$")
  return name
end
local root = dirname(arg[0])

local function get_timestamp(filename)
  local f = io.popen(table.concat{"find ", root, filename,
                                  " -printf '%h/%f %T@\n' 2> /dev/null ",
                                  "| cut -d' ' -f 2" })
  return tonumber(f:read("*l")) or 0
end
local script_timestamp = get_timestamp(basename(arg[0]))

local matrix_methods = {
  float = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda", "same_dim",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "isfinite", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "uniformf", "linspace", "logspace",
    "linear", "sliding_window", "is_contiguous",
    "prune_subnormal_and_check_normal", "adjust_range", "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp", "add", "scalar_add",
    "sub", "mul", "cmul", "plogp", "log", "log1p", "exp", "expm1", "sqrt", "pow", "tan",
    "tanh", "atan", "atanh", "sin", "sinh", "asin", "asinh", "cos", "cosh",
    "acos", "acosh", "abs", "complement", "sign", "sum", "copy", "axpy", "prod",
    "cumsum", "cumprod",
    "gemm", "sparse_mm", "gemv", "ger", "dot", "scal", "cinv",
    "masked_fill", "masked_copy", "div", "norm2", "inv", "logdet", "det",
    "cholesky", "svd", "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to", "index",
    "indexed_fill", "indexed_copy",
    "ceil", "floor", "round", "left_inflate", "right_inflate",
    "count_eq", "count_neq",
  },
  double = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda", "same_dim",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "isfinite", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "uniformf", "linspace", "logspace",
    "linear", "sliding_window", "is_contiguous",
    "prune_subnormal_and_check_normal", "adjust_range", "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp", "add", "scalar_add",
    "sub", "mul", "cmul", "plogp", "log", "log1p", "exp", "expm1", "sqrt", "pow", "tan",
    "tanh", "atan", "atanh", "sin", "sinh", "asin", "asinh", "cos", "cosh",
    "acos", "acosh", "abs", "complement", "sign", "sum", "copy", "axpy", "prod",
    "cumsum", "cumprod",
    "gemm", "gemv", "ger", "dot", "scal", "cinv",
    "masked_fill", "masked_copy", "div", "norm2",
    "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to", "index",
    "indexed_fill", "indexed_copy",
    "ceil", "floor", "round", "left_inflate", "right_inflate",
    "count_eq", "count_neq",
  },
  int32_t = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda", "same_dim",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding", "uniform", "linspace",
    "linear", "sliding_window", "is_contiguous",
    "diag", "fill",
    "zeros", "ones", "min", "max", "equals", "clamp", "add", "scalar_add",
    "sub", "cmul", "scal",
    "sum", "copy", "axpy", "prod",
    "cumsum", "cumprod",
    "masked_fill", "masked_copy", "idiv", "mod",
    "lt", "gt", "eq", "neq", "toMMap",
    "data", "order", "order_rank", "convert_to", "index",
    "indexed_fill", "indexed_copy",
    "left_inflate", "right_inflate",
    "count_eq", "count_neq",
  },
  ComplexF = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda", "same_dim",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "isfinite", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding",
    "linear", "sliding_window", "is_contiguous",
    "prune_subnormal_and_check_normal", "diag", "fill",
    "zeros", "ones", "equals", "add", "scalar_add",
    "sub", "mul", "cmul",
    "sum", "copy", "axpy", "prod",
    "cumsum", "cumprod",
    "gemm", "gemv", "ger", "dot", "scal", "cinv",
    "masked_fill", "masked_copy", "div", "norm2",
    "toMMap",
    "data", "convert_to", "index",
    "indexed_fill", "indexed_copy",
    "ceil", "floor", "round", "left_inflate", "right_inflate",
    "count_eq", "count_neq",
  },
  char = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda", "same_dim",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding",
    "sliding_window", "is_contiguous",
    "diag", "fill",
    "zeros", "ones", "equals",
    "copy",
    "toMMap",
    "data", "convert_to", "index",
    "indexed_fill", "indexed_copy",
    "stringfy",
    "left_inflate", "right_inflate",
    "count_eq", "count_neq",
  },
  bool = {
    "size", "rewrap", "squeeze", "get_reference_string", "copy_from_table",
    "get", "set", "offset", "raw_get", "raw_set", "get_use_cuda", "same_dim",
    "set_use_cuda", "dim", "num_dim", "stride", "slice", "select", "clone",
    "transpose", "toTable", "contiguous", "map", "diagonalize",
    "get_shared_count", "reset_shared_count", "add_to_shared_count", "sync",
    "padding_all", "padding",
    "sliding_window", "is_contiguous",
    "diag", "fill",
    "zeros", "ones", "equals",
    "copy",
    "toMMap",
    "data", "convert_to", "index",
    "indexed_fill", "indexed_copy",
    "to_index",
    "left_inflate", "right_inflate",
  },
}

local matrix_class_methods = {
  float = {
    "as", "deserialize", "read", "fromMMap", "__broadcast__",
    "__call_function__", "__newindex_function__", "__index_function__",
  },
  double = {
    "as", "deserialize", "read", "fromMMap", "__broadcast__",
    "__call_function__", "__newindex_function__", "__index_function__",
  },
  char = {
    "as", "deserialize", "read", "fromMMap", "__broadcast__",
    "__call_function__", "__newindex_function__", "__index_function__",
  },
  int32_t = {
    "as", "deserialize", "read", "fromMMap", "__broadcast__",
    "__call_function__", "__newindex_function__", "__index_function__",
  },
  ComplexF = {
    "as", "deserialize", "read", "fromMMap", "__broadcast__",
    "__call_function__", "__newindex_function__", "__index_function__",
  },
  bool = {
    "as", "deserialize", "read", "fromMMap", "__broadcast__",
    "__call_function__", "__newindex_function__", "__index_function__",
  },
  
}

local matrix_class_method_binding = [[
//BIND_CLASS_METHOD %s %s
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<%s>::%s(L));
}
//BIND_END

]]

local matrix_method_binding = [[
//BIND_METHOD %s %s
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<%s>::%s(L,obj));
}
//BIND_END

]]

local function generate_binding(data,
                                class_methods, methods,
                                class_method_binding, method_binding)
  for _,arg in ipairs(data) do

    local T          = arg[1]
    local MATRIX_T   = arg[2]
    local MATRIX_Lua = arg[3]
    local input      = arg[4]
    local output     = arg[5]

    local input_timestamp  = get_timestamp(input)
    local output_timestamp = get_timestamp(output)
    if input_timestamp > output_timestamp or script_timestamp > output_timestamp then
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
        assert(not generated[name], string.format("Duplicated entry %s.%s", MATRIX_Lua, name))
        f:write((method_binding:format(MATRIX_T, name, T, name)))
        generated[name] = true
      end
      f:close()
    end
  end
end

local matrix_binding_data = {
  { "float", "MatrixFloat", "matrix", "matrix_template.lua.cc", "bind_matrix.lua.cc" },
  { "double", "MatrixDouble", "matrixDouble", "matrix_template.lua.cc", "bind_matrix_double.lua.cc" },
  { "int32_t", "MatrixInt32", "matrixInt32", "matrix_template.lua.cc", "bind_matrix_int32.lua.cc" },
  { "char", "MatrixChar", "matrixChar", "matrix_template.lua.cc", "bind_matrix_char.lua.cc" },
  { "ComplexF", "MatrixComplexF", "matrixComplex", "matrix_template.lua.cc", "bind_matrix_complex_float.lua.cc" },
  { "bool", "MatrixBool", "matrixBool", "matrix_template.lua.cc", "bind_matrix_bool.lua.cc" },
}

generate_binding(matrix_binding_data,
                 matrix_class_methods, matrix_methods,
                 matrix_class_method_binding, matrix_method_binding)
