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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixDouble);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixDouble);

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

typedef MatrixDouble::sliding_window SlidingWindowMatrixDouble;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixDouble);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixDouble matrixDouble
//BIND_CPP_CLASS MatrixDouble
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixDouble Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixDouble matrixDouble.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixDouble

//BIND_CONSTRUCTOR SlidingWindowMatrixDouble
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixDouble
//DOC_BEGIN
// matrixDouble(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::constructor(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::as(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::fromMMap(L));
}
//BIND_END

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixDouble size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble isfinite
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::isfinite(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble uniform
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::uniform(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble uniformf
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::uniformf(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble linspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::linspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble logspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::logspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble linear
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::linear(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble prune_subnormal_and_check_normal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::prune_subnormal_and_check_normal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble adjust_range
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::adjust_range(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble min
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::min(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble max
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::max(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble clamp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::clamp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble scalar_add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::scalar_add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sub
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sub(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble mul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::mul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble cmul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::cmul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble plogp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::plogp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble log
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::log(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble log1p
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::log1p(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble exp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::exp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sqrt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sqrt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble pow
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::pow(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble tan
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::tan(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble tanh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::tanh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble atan
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::atan(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble atanh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::atanh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sin
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sin(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sinh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sinh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble asin
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::asin(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble asinh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::asinh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble cos
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::cos(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble cosh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::cosh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble acos
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::acos(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble acosh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::acosh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble abs
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::abs(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble complement
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::complement(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sign
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sign(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble sum
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::sum(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble axpy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::axpy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble gemm
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::gemm(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble gemv
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::gemv(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble ger
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::ger(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble dot
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::dot(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble scal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::scal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble masked_fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::masked_fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble masked_copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::masked_copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble div
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::div(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble norm2
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::norm2(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble lt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::lt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble gt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::gt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble eq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::eq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble neq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::neq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::data(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble order
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::order(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble order_rank
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::order_rank(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::convert_to(L,obj));
}
//BIND_END

