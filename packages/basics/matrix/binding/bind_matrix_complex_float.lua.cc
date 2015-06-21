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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixComplexF);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixComplexF);

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

typedef MatrixComplexF::sliding_window SlidingWindowMatrixComplexF;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixComplexF);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixComplexF matrixComplex
//BIND_CPP_CLASS MatrixComplexF
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixComplexF Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixComplexF matrixComplex.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixComplexF

//BIND_CONSTRUCTOR SlidingWindowMatrixComplexF
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplexF iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixComplexF
//DOC_BEGIN
// matrixComplex(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::constructor(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixComplexF as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::as(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixComplexF deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixComplexF read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixComplexF fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::fromMMap(L));
}
//BIND_END

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixComplexF size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF isfinite
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::isfinite(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF linear
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::linear(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF prune_subnormal_and_check_normal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::prune_subnormal_and_check_normal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF scalar_add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::scalar_add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF sub
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sub(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF mul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::mul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF cmul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::cmul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF sum
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sum(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF axpy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::axpy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF gemm
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::gemm(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF gemv
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::gemv(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF ger
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::ger(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF dot
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::dot(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF scal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::scal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF masked_fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::masked_fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF masked_copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::masked_copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF div
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::div(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF norm2
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::norm2(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::data(L,obj));
}
//BIND_END

