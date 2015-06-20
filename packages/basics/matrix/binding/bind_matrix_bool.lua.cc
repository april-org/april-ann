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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixBool);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixBool);

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

typedef MatrixBool::sliding_window SlidingWindowMatrixBool;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixBool);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixBool matrixBool
//BIND_CPP_CLASS MatrixBool
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixBool Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixBool matrixBool.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixBool

//BIND_CONSTRUCTOR SlidingWindowMatrixBool
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// matrixBool(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::constructor(L));
}
//BIND_END

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixBool size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool adjust_range
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::adjust_range(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::data(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::convert_to(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool to_index
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::to_index(L,obj));
}
//BIND_END

