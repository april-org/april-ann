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

using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Boolean;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilMath::MatrixExt::LAPACK;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixComplex);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixComplex);

//BIND_END

//BIND_HEADER_H
#include "bind_april_io.h"
#include "bind_mtrand.h"
#include "gpu_mirrored_memory_block.h"
#include "matrixFloat.h"
#include "luabindmacros.h"
#include "luabindutil.h"
#include "utilLua.h"

using namespace Basics;

typedef MatrixComplex::sliding_window SlidingWindowMatrixComplex;

DECLARE_LUA_TABLE_BIND_SPECIALIZATION(SlidingWindowMatrixComplex;);

#include "matrix_binding.h"

//BIND_END

//BIND_LUACLASSNAME MatrixComplex matrixComplex
//BIND_CPP_CLASS MatrixComplex
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixComplex Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixComplex matrixComplex.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixComplex

//BIND_CONSTRUCTOR SlidingWindowMatrixComplex
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex get_matrix
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_matrix(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex next
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::next(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex set_at_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::set_at_window(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex num_windows
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::num_windows(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex coords
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::coords(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex is_end
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::is_end(L, obj));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixComplex iterate
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::iterate(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
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

///////////////////////////////////////////////////////////

//BIND_METHOD MatrixComplex size
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::size(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex squeeze
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::squeeze(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_reference_string(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex copy_from_table
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex offset
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::offset(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex get_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex num_dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::num_dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex clone
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::clone(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex isfinite
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::isfinite(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex toTable
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::toTable(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::map(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex diagonalize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::diagonalize(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex get_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::get_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex reset_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::reset_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sync
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sync(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex uniform
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::uniform(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex uniformf
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::uniformf(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex linspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::linspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex logspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::logspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex linear
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::linear(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sliding_window
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sliding_window(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex is_contiguous
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::is_contiguous(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex prune_subnormal_and_check_normal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::prune_subnormal_and_check_normal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex adjust_range
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::adjust_range(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex diag
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::diag(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex zeros
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::zeros(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex ones
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::ones(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex min
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::min(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex max
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::max(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex clamp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::clamp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex scalar_add
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::scalar_add(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sub
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sub(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex mul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::mul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex cmul
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::cmul(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex plogp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::plogp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex log
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::log(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex log1p
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::log1p(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex exp
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::exp(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sqrt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sqrt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex pow
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::pow(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex tan
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::tan(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex tanh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::tanh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex atan
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::atan(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex atanh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::atanh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sin
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sin(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sinh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sinh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex asin
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::asin(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex asinh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::asinh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex cos
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::cos(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex cosh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::cosh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex acos
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::acos(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex acosh
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::acosh(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex abs
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::abs(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex complement
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::complement(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sign
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sign(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex sum
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::sum(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex axpy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::axpy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex gemm
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::gemm(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex gemv
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::gemv(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex ger
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::ger(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex dot
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::dot(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex scal
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::scal(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex masked_fill
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::masked_fill(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex masked_copy
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::masked_copy(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex div
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::div(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex norm2
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::norm2(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex lt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::lt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex gt
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::gt(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex eq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::eq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex neq
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::neq(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::toMMap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex data
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::data(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex order
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::order(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex order_rank
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::order_rank(L,obj));
}
//BIND_END

//BIND_METHOD MatrixComplex convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<ComplexF>::convert_to(L,obj));
}
//BIND_END

