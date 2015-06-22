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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixFloat);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixFloat);

//BIND_END

//BIND_HEADER_H
#include "complex_number.h"
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

typedef MatrixFloat::sliding_window SlidingWindowMatrixFloat;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixFloat);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixFloat matrix
//BIND_CPP_CLASS MatrixFloat
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixFloat Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixFloat matrix.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixFloat

//BIND_CONSTRUCTOR SlidingWindowMatrixFloat
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixFloat iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// matrix(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::constructor(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::as(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::fromMMap(L));
}
//BIND_END

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixFloat size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat isfinite
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::isfinite(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat uniform
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::uniform(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat uniformf
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::uniformf(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat linspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::linspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat logspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::logspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat linear
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::linear(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat prune_subnormal_and_check_normal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::prune_subnormal_and_check_normal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat adjust_range
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::adjust_range(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat min
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::min(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat max
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::max(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat clamp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::clamp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat scalar_add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::scalar_add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sub
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sub(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat mul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::mul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cmul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::cmul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat plogp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::plogp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat log
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::log(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat log1p
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::log1p(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat exp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::exp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sqrt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sqrt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat pow
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::pow(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat tan
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::tan(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat tanh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::tanh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat atan
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::atan(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat atanh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::atanh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sin
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sin(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sinh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sinh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat asin
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::asin(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat asinh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::asinh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cos
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::cos(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cosh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::cosh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat acos
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::acos(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat acosh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::acosh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat abs
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::abs(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat complement
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::complement(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sign
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sign(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sum
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sum(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat axpy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::axpy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat gemm
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::gemm(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sparse_mm
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::sparse_mm(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat gemv
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::gemv(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat ger
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::ger(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat dot
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::dot(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat scal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::scal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat masked_fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::masked_fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat masked_copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::masked_copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat div
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::div(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat norm2
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::norm2(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat inv
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::inv(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat logdet
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::logdet(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat det
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::det(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cholesky
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::cholesky(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat svd
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::svd(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat lt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::lt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat gt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::gt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat eq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::eq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat neq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::neq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::data(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat order
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::order(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat order_rank
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::order_rank(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::convert_to(L,obj));
}
//BIND_END

