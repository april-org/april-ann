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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixInt32);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixInt32);

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

typedef MatrixInt32::sliding_window SlidingWindowMatrixInt32;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixInt32);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixInt32 matrixInt32
//BIND_CPP_CLASS MatrixInt32
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixInt32 Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixInt32 matrixInt32.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixInt32

//BIND_CONSTRUCTOR SlidingWindowMatrixInt32
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// matrixInt32(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::constructor(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixInt32 as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::as(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixInt32 deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixInt32 read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixInt32 fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::fromMMap(L));
}
//BIND_END

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixInt32 size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 uniform
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::uniform(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 linspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::linspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 linear
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::linear(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 min
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::min(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 max
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::max(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 clamp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::clamp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 masked_fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::masked_fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 masked_copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::masked_copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 lt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::lt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 gt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::gt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 eq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::eq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 neq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::neq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::data(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 order
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::order(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 order_rank
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::order_rank(L,obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<int32_t>::convert_to(L,obj));
}
//BIND_END

