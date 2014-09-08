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
#include "activation_function_kernels.h"
#include "april_assert.h"
#include "cblas_headers.h"
#include "cmath_overloads.h"
#include "map_matrix.h"
#include "smart_ptr.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt;
using namespace AprilMath::MatrixExt::Operations;

#ifdef USE_CUDA
#include "cuda_utils.h"
#include "ceiling_power_of_two.h"
#include "gpu_helper.h"

namespace AprilMath {
  
  static __device__ unsigned int getMatrixIndex(unsigned int x,
                                                unsigned int lda_x,
                                                unsigned int y) {
    return x*lda_x + y;
  }

  __global__ void minMaxFirstReduction(const float *input_units,
                                       float *minimums,
                                       float *maximums,
                                       unsigned int reduction_top,
                                       unsigned int max_x,
                                       unsigned int max_y,
                                       unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    unsigned int active_reduction = reduction_top >> 1;

    if (matrix_y_pos < max_y) {
      if (matrix_x_pos < active_reduction) {
        unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
        unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction),
                                                    lda_x, matrix_y_pos);

        if (matrix_x_pos + active_reduction < max_x) {
          if (input_units[index] > input_units[passive_index]) {
            minimums[index] = input_units[passive_index];
            maximums[index] = input_units[index];
          }
          else {
            minimums[index] = input_units[index];
            maximums[index] = input_units[passive_index];
          }
        }
        else {
          minimums[index] = input_units[index];
          maximums[index] = input_units[index];
        }
      }
    }
  }

  __global__ void minMaxNextReduction(float *minimums,
                                      float *maximums,
                                      unsigned int reduction_top,
                                      unsigned int max_y,
                                      unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    unsigned int active_reduction = reduction_top >> 1;

    if (matrix_y_pos < max_y) {
      if (matrix_x_pos < active_reduction) {
        unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
        unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction),
                                                    lda_x, matrix_y_pos);

        if (minimums[index] > minimums[passive_index])
          minimums[index] = minimums[passive_index];
        // else we do not modify anything
        if (maximums[index] < maximums[passive_index])
          maximums[index] = maximums[passive_index];
      }
    }
  }

  __global__ void sumFirstReduction(const float *output_units,
                                    float *sums,
                                    unsigned int reduction_top,
                                    unsigned int max_x,
                                    unsigned int max_y,
                                    unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    unsigned int active_reduction = reduction_top >> 1;

    if (matrix_y_pos < max_y) {
      if (matrix_x_pos < active_reduction) {
        unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
        unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction),
                                                    lda_x, matrix_y_pos);

        if (matrix_x_pos + active_reduction < max_x)
          sums[index] = output_units[index] + output_units[passive_index];
        else
          sums[index] = output_units[index];
      }
    }
  }

  __global__ void sumNextReduction(float *sums,
                                   unsigned int reduction_top,
                                   unsigned int max_y,
                                   unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    unsigned int active_reduction = reduction_top >> 1;

    if (matrix_y_pos < max_y) {
      if (matrix_x_pos < active_reduction) {
        unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
        unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction),
                                                    lda_x, matrix_y_pos);

        sums[index] = sums[index] + sums[passive_index];
      }
    }
  }

  __global__ void applyMinimumNorm(float *minimums,
                                   float *maximums,
                                   unsigned int max_x) {
    unsigned int matrix_x_pos = CUDA::getArrayIndex(blockIdx, blockDim, threadIdx);
    if (matrix_x_pos < max_x)
      if ((maximums[matrix_x_pos] - minimums[matrix_x_pos]) > 30.0f)
        minimums[matrix_x_pos] = maximums[matrix_x_pos] - 30.0f;
  }

  __global__ void applyExpMinus(const float *input_units,
                                float *output_units,
                                float *data,
                                unsigned int max_x,
                                unsigned int max_y,
                                unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
      unsigned int index  = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      output_units[index] = AprilMath::m_exp(input_units[index] - data[matrix_y_pos]);
    }
  }

  __global__ void applyMinusLog(const float *input_units,
                                float *output_units,
                                float *sums,
                                float *maximums,
                                unsigned int max_x,
                                unsigned int max_y,
                                unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
      unsigned int index  = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      output_units[index] = (input_units[index] -
                             maximums[matrix_y_pos] -
                             AprilMath::m_log(sums[matrix_y_pos]));
    }
  }

  __global__ void applyInverse(float *sums,
                               unsigned int max_x) {
    unsigned int matrix_x_pos = CUDA::getArrayIndex(blockIdx, blockDim, threadIdx);
    if (matrix_x_pos < max_x)
      sums[matrix_x_pos] = 1.0f/sums[matrix_x_pos];
  }



  __global__ void applyRatio(float *output_units,
                             float *ratios,
                             unsigned int max_x,
                             unsigned int max_y,
                             unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    CUDA::getMatrixIndices(blockIdx,
                           blockDim,
                           threadIdx,
                           matrix_x_pos,
                           matrix_y_pos);
    if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
      unsigned int index  = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      output_units[index] = ratios[matrix_y_pos] * output_units[index];   
    }
  }

} // namespace AprilMath
#endif

namespace ANN {
  namespace Kernels {
    
    void applyHardTanhDerivative(Basics::MatrixFloat *output_errors,
                                 Basics::MatrixFloat *input_units,
                                 float inf, float sup) {
      MatrixScalarMap1(input_units,
                       AprilMath::m_curried_clamp_der<float>(inf, sup),
                       output_errors);
    }

    void applyTanh(Basics::MatrixFloat *output,
                   Basics::MatrixFloat *input) {
      MatrixScalarMap1(input,
                       AprilMath::Functors::m_antisym_logistic<float>(),
                       output);
    }
    
    void applyTanhDerivative(Basics::MatrixFloat *output_errors,
                             Basics::MatrixFloat *output_units) {
      MatrixScalarMap1(output_units,
                       AprilMath::Functors::m_antisym_logistic_der<float>(),
                       output_errors);
    }

    
    void applyLogistic(Basics::MatrixFloat *output,
                       Basics::MatrixFloat *input) {
      MatrixScalarMap1(input,
                       AprilMath::Functors::m_logistic<float>(),
                       output);
    }
    
    void applyLogisticDerivative(Basics::MatrixFloat *output_errors,
                                 Basics::MatrixFloat *output_units) {
      MatrixScalarMap1(output_units,
                       AprilMath::Functors::m_logistic_der<float>(),
                       output_errors);
    }
    
    void applySoftsign(Basics::MatrixFloat *output,
                       Basics::MatrixFloat *input) {
      MatrixScalarMap1(input,
                       AprilMath::Functors::m_softsign<float>(),
                       output);
    }
    
    void applySoftsignDerivative(Basics::MatrixFloat *output_errors,
                                 Basics::MatrixFloat *output_units) {
      MatrixScalarMap1(output_units,
                       AprilMath::Functors::m_softsign_der<float>(),
                       output_errors);
    }

    void applySoftplus(Basics::MatrixFloat *output,
                       Basics::MatrixFloat *input) {
      MatrixScalarMap1(input,
                       AprilMath::Functors::m_softplus<float>(),
                       output);
    }
    
    void applySoftplusDerivative(Basics::MatrixFloat *output_errors,
                                 Basics::MatrixFloat *input_units) {
      MatrixScalarMap1(input_units,
                       AprilMath::Functors::m_softplus_der<float>(),
                       output_errors);
    }

    void applyReLU(Basics::MatrixFloat *output,
                   Basics::MatrixFloat *input) {
      MatrixScalarMap1(input,
                       AprilMath::Functors::m_relu<float>(),
                       output);
    }
    
    void applyReLUDerivative(Basics::MatrixFloat *output_errors,
                             Basics::MatrixFloat *input_units) {
      MatrixScalarMap1(input_units,
                       AprilMath::Functors::m_relu_der<float>(),
                       output_errors);
    }

    void applyLogLogistic(Basics::MatrixFloat *output,
                          Basics::MatrixFloat *input) {
      MatrixScalarMap1(input,
                       AprilMath::Functors::m_log_logistic<float>(),
                       output);
    }

    void applySoftmax(Basics::MatrixFloat *output,
                      Basics::MatrixFloat *input) {
      unsigned int bunch_size = input->getDimSize(0);
      unsigned int size = input->getDimSize(1);
      AprilMath::FloatGPUMirroredMemoryBlock *input_units =
        input->getRawDataAccess();
      AprilMath::FloatGPUMirroredMemoryBlock *output_units =
        output->getRawDataAccess();
#ifdef USE_CUDA
      if (input->getCudaFlag()) {
        unsigned int reduction_size = AprilUtils::ceilingPowerOfTwo(size) >> 1;
        unsigned int mem_size = reduction_size * bunch_size;
        AprilUtils::SharedPtr<FloatGPUMirroredMemoryBlock> minimums =
          new FloatGPUMirroredMemoryBlock(mem_size);
        AprilUtils::SharedPtr<FloatGPUMirroredMemoryBlock> maximums =
          new FloatGPUMirroredMemoryBlock(mem_size);
        AprilUtils::SharedPtr<FloatGPUMirroredMemoryBlock> sums =
          new FloatGPUMirroredMemoryBlock(mem_size);
        //
        const float *input_units_ptr = input_units->getGPUForRead();
        float *output_units_ptr      = output_units->getGPUForWrite();
        float *minimums_ptr          = minimums->getGPUForWrite();
        float *maximums_ptr          = maximums->getGPUForWrite();
        float *sums_ptr              = sums->getGPUForWrite();
        unsigned int units_top       = AprilUtils::ceilingPowerOfTwo(size);
        unsigned int top_reduction   = units_top;
        dim3 block, grid;
        CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                                  block, grid);
        
        minMaxFirstReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (input_units_ptr,
           minimums_ptr,
           maximums_ptr,
           top_reduction,
           size,
           bunch_size,
           bunch_size);
        for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
          CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                                    block, grid);
          minMaxNextReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
            (minimums_ptr,
             maximums_ptr,
             top_reduction,
             bunch_size,
             bunch_size);
        }
        
        CUDA::computeBlockAndGridSizesForArray(bunch_size, block, grid);
        applyMinimumNorm<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (minimums_ptr,
           maximums_ptr,
           bunch_size);
        
        CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                                  block, grid);
        
        applyExpMinus<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (input_units_ptr,
           output_units_ptr,
           minimums_ptr,
           size,
           bunch_size,
           bunch_size);
        
        // We reset the top
        top_reduction = units_top;
        
        sumFirstReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (output_units_ptr,
           sums_ptr,
           top_reduction,
           size,
           bunch_size,
           bunch_size);
        for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
          CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                                    block, grid);
          sumNextReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
            (sums_ptr,
             top_reduction,
             bunch_size,
             bunch_size);
        }
        
        CUDA::computeBlockAndGridSizesForArray(bunch_size, block, grid);
        applyInverse<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (sums_ptr,
           bunch_size);
        
        CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                                  block, grid);
        
        applyRatio<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (output_units_ptr,
           sums_ptr,
           size,
           bunch_size,
           bunch_size);
      }
      else {
#endif
        const float *input_units_ptr = input_units->getPPALForRead();
        float *output_units_ptr      = output_units->getPPALForWrite();

        for (unsigned int b = 0; b < bunch_size; ++b) {
          float minimum = input_units_ptr[0];
          float maximum = input_units_ptr[0];
          unsigned int cur_pos = bunch_size;
          for (unsigned int i = 2; i < size; i += 2) {
            float prev_unit = input_units_ptr[cur_pos];
            cur_pos += bunch_size;
            float cur_unit = input_units_ptr[cur_pos];
            cur_pos += bunch_size;
            if (prev_unit < cur_unit) {
              if (prev_unit < minimum) minimum = prev_unit;
              if (cur_unit > maximum) maximum = cur_unit;
            } else {
              if (cur_unit < minimum) minimum = cur_unit;
              if (prev_unit > maximum) maximum = prev_unit;
            }
          }
          if ((size & 1) == 0) { // si es impar
            unsigned int last_pos = (size - 1) * bunch_size;
            if (input_units_ptr[last_pos] < minimum)
              minimum = input_units_ptr[last_pos];
            if (input_units_ptr[last_pos] > maximum)
              maximum = input_units_ptr[last_pos];
          }
          if ((maximum - minimum) > 30.0f) minimum = maximum - 30.0f;
          double addition = 0;
          cur_pos = 0;
          for (unsigned int i = 0; i < size; i++) {
            double e = exp(input_units_ptr[cur_pos] - minimum);
            output_units_ptr[cur_pos] = e;
            addition += e;
            cur_pos  += bunch_size;
          }
          float ratio = 1.0f/addition;
          cblas_sscal(size, ratio, output_units_ptr, bunch_size);
          output_units_ptr++;
          input_units_ptr++;
        }
#ifdef USE_CUDA
      }
#endif
    }

    void applyLogSoftmax(Basics::MatrixFloat *output,
                         Basics::MatrixFloat *input) {
      unsigned int bunch_size = input->getDimSize(0);
      unsigned int size = input->getDimSize(1);
      AprilMath::FloatGPUMirroredMemoryBlock *input_units =
        input->getRawDataAccess();
      AprilMath::FloatGPUMirroredMemoryBlock *output_units =
        output->getRawDataAccess();
#ifdef USE_CUDA
      if (input->getCudaFlag()) {
        unsigned int reduction_size = AprilUtils::ceilingPowerOfTwo(size) >> 1;
        unsigned int mem_size = reduction_size * bunch_size;
        AprilUtils::SharedPtr<FloatGPUMirroredMemoryBlock> minimums =
          new FloatGPUMirroredMemoryBlock(mem_size);
        AprilUtils::SharedPtr<FloatGPUMirroredMemoryBlock> maximums =
          new FloatGPUMirroredMemoryBlock(mem_size);
        AprilUtils::SharedPtr<FloatGPUMirroredMemoryBlock> sums =
          new FloatGPUMirroredMemoryBlock(mem_size);
        //
        const float *input_units_ptr = input_units->getGPUForRead();
        float *output_units_ptr      = output_units->getGPUForWrite();
        float *minimums_ptr          = minimums->getGPUForWrite();
        float *maximums_ptr          = maximums->getGPUForWrite();
        float *sums_ptr              = sums->getGPUForWrite();
        unsigned int units_top       = AprilUtils::ceilingPowerOfTwo(size);
        unsigned int top_reduction   = units_top;
        dim3 block, grid;
        CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                                  block, grid);
      
        minMaxFirstReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (input_units_ptr,
           minimums_ptr,
           maximums_ptr,
           top_reduction,
           size,
           bunch_size,
           bunch_size);
        for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
          CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                                    block, grid);
          minMaxNextReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
            (minimums_ptr,
             maximums_ptr,
             top_reduction,
             bunch_size,
             bunch_size);
        }
      
        CUDA::computeBlockAndGridSizesForArray(bunch_size, block, grid);
        applyMinimumNorm<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (minimums_ptr,
           maximums_ptr,
           bunch_size);

        CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                                  block, grid);

        applyExpMinus<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (input_units_ptr,
           output_units_ptr,
           maximums_ptr,
           size,
           bunch_size,
           bunch_size);
    
        // We reset the top
        top_reduction = units_top;

        sumFirstReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (output_units_ptr,
           sums_ptr,
           top_reduction,
           size,
           bunch_size,
           bunch_size);
        for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
          CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                                    block, grid);
          sumNextReduction<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
            (sums_ptr,
             top_reduction,
             bunch_size,
             bunch_size);
        }
      
        CUDA::computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                                  block, grid);

        applyMinusLog<<<grid, block, 0, CUDA::GPUHelper::getCurrentStream()>>>
          (input_units_ptr,
           output_units_ptr,
           sums_ptr,
           maximums_ptr,
           size,
           bunch_size,
           bunch_size);
      }
      else {
#endif
        const float *input_units_ptr = input_units->getPPALForRead();
        float *output_units_ptr      = output_units->getPPALForWrite();

        for (unsigned int b = 0; b < bunch_size; ++b) {
          float maximum = input_units_ptr[0];
          unsigned int cur_pos = bunch_size;
          for (unsigned int i = 2; i < size; i += 2) {
            float prev_unit = input_units_ptr[cur_pos];
            cur_pos += bunch_size;
            float cur_unit = input_units_ptr[cur_pos];
            if (prev_unit < cur_unit) {
              if (cur_unit > maximum) maximum = cur_unit;
            } else {
              if (prev_unit > maximum) maximum = prev_unit;
            }
            cur_pos += bunch_size;
          }
          if ((size & 1) == 0) { // si es par
            unsigned int last_pos = (size - 1) * bunch_size;
            if (input_units_ptr[last_pos] > maximum)
              maximum = input_units_ptr[last_pos];
          }
          double addition = 0.0f;
          cur_pos = 0;
          for (unsigned int i = 0; i < size; i++) {
            output_units_ptr[cur_pos] = input_units_ptr[cur_pos] - maximum;
            double exp_output = AprilMath::m_exp(static_cast<double>(output_units_ptr[cur_pos]));
            addition += exp_output;
            cur_pos  += bunch_size;
          }
          float ratio = static_cast<float>(log(addition));
          cur_pos = 0;
          for (unsigned int i = 0; i < size; i++) {
            output_units_ptr[cur_pos] -= ratio;
            april_assert(!(output_units_ptr[cur_pos] > 0.0f) &&
                         "Numerical inestability at log-softmax activation function");
            cur_pos += bunch_size;
          }
          output_units_ptr++;
          input_units_ptr++;
        }
#ifdef USE_CUDA
      }
#endif      
    }
    
    void applySoftmaxDerivative(Basics::MatrixFloat *output_errors_mat,
                                Basics::MatrixFloat *input_errors_mat,
                                Basics::MatrixFloat *output_units_mat) {
      unsigned int bunch_size = output_units_mat->getDimSize(0);
      unsigned int size = output_units_mat->getDimSize(1);
      AprilMath::FloatGPUMirroredMemoryBlock *output_errors =
        output_errors_mat->getRawDataAccess();
      AprilMath::FloatGPUMirroredMemoryBlock *input_errors =
        input_errors_mat->getRawDataAccess();
      AprilMath::FloatGPUMirroredMemoryBlock *output_units =
        output_units_mat->getRawDataAccess();
#ifdef USE_CUDA
      if (output_units->getCudaFlag()) {
        ERROR_PRINT("NOT IMPLEMENTED FOR CUDA!!!\n");
      }
      // else {
#endif
      const float *output_units_ptr = output_units->getPPALForRead();
      const float *input_errors_ptr = input_errors->getPPALForRead();
      float *output_errors_ptr      = output_errors->getPPALForWrite();
      
      for (unsigned int b = 0; b < bunch_size; ++b) {
        float sum = 0.0f;
        unsigned int cur_pos = 0;
        for (unsigned int i = 0; i < size; i++) {
          sum += output_units_ptr[cur_pos] * input_errors_ptr[cur_pos];
          cur_pos += bunch_size;
        }
        cur_pos = 0;
        for (unsigned int i = 0; i < size; i++) {
          output_errors_ptr[cur_pos] = ( output_units_ptr[cur_pos] *
                                         ( input_errors_ptr[cur_pos] - sum ) );
          cur_pos += bunch_size;
        }
        output_units_ptr++;
        input_errors_ptr++;
        output_errors_ptr++;
      }
#ifdef USE_CUDA
      //  }
#endif
    }
    }
}
