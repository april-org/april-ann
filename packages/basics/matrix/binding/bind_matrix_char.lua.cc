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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixChar);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixChar);

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

typedef MatrixChar::sliding_window SlidingWindowMatrixChar;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixChar);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixChar matrixChar
//BIND_CPP_CLASS MatrixChar
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixChar Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixChar matrixChar.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixChar

//BIND_CONSTRUCTOR SlidingWindowMatrixChar
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// matrixChar(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::constructor(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixChar as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::as(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixChar deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixChar read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixChar fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::fromMMap(L));
}
//BIND_END

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixChar size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar adjust_range
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::adjust_range(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar clamp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::clamp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar masked_fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::masked_fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar masked_copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::masked_copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar lt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::lt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar gt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::gt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar eq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::eq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar neq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::neq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::data(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar order
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::order(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar order_rank
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::order_rank(L,obj));
}
//BIND_END

//BIND_METHOD MatrixChar convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<char>::convert_to(L,obj));
}
//BIND_END

