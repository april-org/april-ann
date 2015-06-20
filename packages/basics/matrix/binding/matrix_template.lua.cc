//BIND_HEADER_C
extern "C" {
#include <ctype.h>
}
#include "bind_april_io.h"
#include "bind_mathcore.h"
#include "bind_mtrand.h"
#include "bind_matrix_bool.h"
#include "bind_matrix_char.h"
#include "bind_matrix_complex_float.h"
#include "bind_matrix_double.h"
#include "bind_matrix_int32.h"
#include "bind_sparse_matrix.h"
#include "luabindutil.h"
#include "luabindmacros.h"
#include "lua_string.h"
#include "matrix_ext.h"
#include "mystring.h"
#include "smart_ptr.h"
#include "sparse_matrixFloat.h"
#include "utilMatrixChar.h"
#include "utilMatrixComplexF.h"
#include "utilMatrixDouble.h"
#include "utilMatrixFloat.h"
#include "utilMatrixInt32.h"

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION($$MATRIX_T$$);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindow$$MATRIX_T$$);

//BIND_END

//BIND_HEADER_H
#include "bind_april_io.h"
#include "bind_mtrand.h"
#include "gpu_mirrored_memory_block.h"
#include "matrixBool.h"
#include "matrixChar.h"
#include "matrixComplexF.h"
#include "matrixInt32.h"
#include "matrixFloat.h"
#include "luabindmacros.h"
#include "luabindutil.h"
#include "utilLua.h"

using namespace Basics;

typedef $$MATRIX_T$$::sliding_window SlidingWindow$$MATRIX_T$$;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindow$$MATRIX_T$$);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME $$MATRIX_T$$ $$MATRIX_Lua$$
//BIND_CPP_CLASS $$MATRIX_T$$
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF $$MATRIX_T$$ Serializable

//BIND_LUACLASSNAME SlidingWindow$$MATRIX_T$$ $$MATRIX_Lua$$.__sliding_window__
//BIND_CPP_CLASS SlidingWindow$$MATRIX_T$$

//BIND_CONSTRUCTOR SlidingWindow$$MATRIX_T$$
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindow$$MATRIX_T$$ iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// $$MATRIX_Lua$$(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<$$T$$>::constructor(L));
}
//BIND_END

