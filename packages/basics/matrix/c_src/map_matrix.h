/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#ifndef MAP_MATRIX_H
#define MAP_MATRIX_H

#define DEFAULT_N_TH 100
#define DEFAULT_SIZE_TH 100u

namespace Basics {
  // forward declaration
  template <typename T>
  class Matrix;
}

namespace AprilMath {

  namespace MatrixExt {

    /**
     * @brief Applies a span-based unary MAP operation.
     *
     * The MAP operation must be a functor with the following header:
     *
     * - <tt>O functor(unsigned int N, const GPUMirroredMemoryBlock<T> *input, unsigned int input_stride, unsigned int input_shift, GPUMirroredMemoryBlock<T> *output, unsigned int output_stride, unsigned int output_shift, bool use_cuda);
     *
     * @tparam T - The type of input Basics::Matrix.
     *
     * @tparam O - The type of output Basics::Matrix.
     *
     * @tparam OP - The type of the functor operator, normally it can be derived
     * from the given functor parameter.
     *
     * @param input - The input Basics::Matrix from where input spans will be
     * taken.
     *
     * @param functor - The functor which computes the MAP operation over a span.
     *
     * @param dest - The output Basics::Matrix. If @c NULL given or not given at
     * all, a new Matrix of type @c O will be allocated with the needed space.
     *
     * @note It is possible that @c input=dest and the computation will be done
     * in-place.
     *
     * @note Uses OMP, if available, to improve the performance.
     */
    template<typename T, typename O, typename OP>
    Basics::Matrix<O> *MatrixSpanMap1(const Basics::Matrix<T> *input,
                                      const OP &functor,
                                      Basics::Matrix<O> *dest = 0,
                                      const int N_th = DEFAULT_N_TH,
                                      const unsigned int SIZE_th = DEFAULT_SIZE_TH);
  
    /**
     * @brief Applies a scalar-based unary MAP operation.
     *
     * The MAP operation must be a functor with the following header:
     *
     * - <tt>O functor(const T &v);
     *
     * @tparam T - The type of input Basics::Matrix.
     *
     * @tparam O - The type of output Basics::Matrix.
     *
     * @tparam OP - The type of the functor operator, normally it can be derived
     * from the given functor parameter.
     *
     * @param input - The input Basics::Matrix.
     *
     * @param functor - The functor which computes the MAP operation over scalars.
     *
     * @param dest - The output Basics::Matrix. If @c NULL given or not given at
     * all, a new Matrix of type @c O will be allocated with the needed space.
     *
     * @note It is possible that @c input=dest and the computation will be done
     * in-place.
     *
     * @note This function is a wrapper over MatrixSpanMap1 which converts a
     * scalar functor into span functor using ScalarToSpanMap1 struct.
     *
     * @note Uses OMP, if available, to improve the performance.
     */
    template<typename T, typename O, typename OP>
    Basics::Matrix<O> *MatrixScalarMap1(const Basics::Matrix<T> *input,
                                        const OP &functor,
                                        Basics::Matrix<O> *dest = 0,
                                        const int N_th = DEFAULT_N_TH,
                                        const unsigned int SIZE_th = DEFAULT_SIZE_TH);
  
    /**
     * @brief Applies a span-based binary MAP operation.
     *
     * The MAP operation must be a functor with the following header:
     *
     * - <tt>O functor(unsigned int N, const GPUMirroredMemoryBlock<T1> *input1, unsigned int input1_stride, unsigned int input1_shift, const GPUMirroredMemoryBlock<T2> *input2, unsigned int input2_stride, unsigned int input2_shift, GPUMirroredMemoryBlock<T> *output, unsigned int output_stride, unsigned int output_shift, bool use_cuda);
     *
     * @tparam T1 - The type of input1 Basics::Matrix.
     *
     * @tparam T2 - The type of input2 Basics::Matrix.
     *
     * @tparam O - The type of output Basics::Matrix.
     *
     * @tparam OP - The type of the functor operator, normally it can be derived
     * from the given functor parameter.
     *
     * @param input1 - The input1 Basics::Matrix from where input spans will be
     * taken.
     *
     * @param input2 - The input2 Basics::Matrix from where input spans will be
     * taken.
     *
     * @param functor - The functor which computes the MAP operation over a span.
     *
     * @param dest - The output Basics::Matrix. If @c NULL given or not given at
     * all, a new Matrix of type @c O will be allocated with the needed space.
     *
     * @note It is possible that @c input1=dest or @c input2=dest and the
     * computation will be done in-place.
     *
     * @note Uses OMP, if available, to improve the performance.
     */
    template<typename T1, typename T2, typename O, typename OP>
    Basics::Matrix<O> *MatrixSpanMap2(const Basics::Matrix<T1> *input1,
                                      const Basics::Matrix<T2> *input2,
                                      const OP &functor,
                                      Basics::Matrix<O> *dest = 0,
                                      const int N_th = DEFAULT_N_TH,
                                      const unsigned int SIZE_th  = DEFAULT_SIZE_TH);

    /**
     * @brief Applies a scalar-based binary MAP operation.
     *
     * The MAP operation must be a functor with the following header:
     *
     * - <tt>O functor(const T1 &a, const T2 &b);
     *
     * @tparam T1 - The type of input1 Basics::Matrix.
     *
     * @tparam T2 - The type of input2 Basics::Matrix.
     *
     * @tparam O - The type of output Basics::Matrix.
     *
     * @tparam OP - The type of the functor operator, normally it can be derived
     * from the given functor parameter.
     *
     * @param input1 - The input1 Basics::Matrix.
     *
     * @param input2 - The input2 Basics::Matrix.
     *
     * @param functor - The functor which computes the MAP operation over scalars.
     *
     * @param dest - The output Basics::Matrix. If @c NULL given or not given at
     * all, a new Matrix of type @c O will be allocated with the needed space.
     *
     * @note It is possible that @c input1=dest or @c input2=dest and the
     * computation will be done in-place.
     *
     * @note This function is a wrapper over MatrixSpanMap1 which converts a
     * scalar functor into span functor using ScalarToSpanMap1 struct.
     *
     * @note Uses OMP, if available, to improve the performance.
     */
    template<typename T1, typename T2, typename O, typename OP>
    Basics::Matrix<O> *MatrixScalarMap2(const Basics::Matrix<T1> *input1,
                                        const Basics::Matrix<T2> *input2,
                                        const OP &functor,
                                        Basics::Matrix<O> *dest = 0,
                                        const int N_th = DEFAULT_N_TH,
                                        const unsigned int SIZE_th = DEFAULT_SIZE_TH);    

  } // namespace MatrixExt

} // namespace AprilMath

#undef DEFAULT_N_TH
#undef DEFAULT_SIZE_TH

#endif // MAP_MATRIX_H
