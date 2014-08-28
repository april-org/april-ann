/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#ifndef NO_OMP
#include <omp.h>
#endif
#include "cmath_overloads.h"
#include "map_template.h"
#include "math_wrappers.h"
#include "unused_variable.h"
#include "wrapper.h"

///////////////////////////////////////////////////////////
/////////////////// Kernels ///////////////////////////////
///////////////////////////////////////////////////////////

#ifdef USE_CUDA
#include "cuda_utils.h"

namespace april_math {

  __global__ void minMaxFirstReduction(const float *input_units,
                                       float *minimums,
                                       float *maximums,
                                       unsigned int reduction_top,
                                       unsigned int max_x,
                                       unsigned int max_y,
                                       unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    getMatrixIndices(blockIdx,
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
    getMatrixIndices(blockIdx,
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
    getMatrixIndices(blockIdx,
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
    getMatrixIndices(blockIdx,
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
    unsigned int matrix_x_pos = getArrayIndex(blockIdx, blockDim, threadIdx);
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
    getMatrixIndices(blockIdx,
                     blockDim,
                     threadIdx,
                     matrix_x_pos,
                     matrix_y_pos);
    if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
      unsigned int index  = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      output_units[index] = expf(input_units[index] - data[matrix_y_pos]);
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
    getMatrixIndices(blockIdx,
                     blockDim,
                     threadIdx,
                     matrix_x_pos,
                     matrix_y_pos);
    if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
      unsigned int index  = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      output_units[index] = (input_units[index] -
                             maximums[matrix_y_pos] - log(sums[matrix_y_pos]));
    }
  }

  __global__ void applyInverse(float *sums,
                               unsigned int max_x) {
    unsigned int matrix_x_pos = getArrayIndex(blockIdx, blockDim, threadIdx);
    if (matrix_x_pos < max_x)
      sums[matrix_x_pos] = 1.0f/sums[matrix_x_pos];
  }



  __global__ void applyRatio(float *output_units,
                             float *ratios,
                             unsigned int max_x,
                             unsigned int max_y,
                             unsigned int lda_x) {
    unsigned int matrix_x_pos, matrix_y_pos;
    getMatrixIndices(blockIdx,
                     blockDim,
                     threadIdx,
                     matrix_x_pos,
                     matrix_y_pos);
    if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
      unsigned int index  = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      output_units[index] = ratios[matrix_y_pos] * output_units[index];   
    }
  }

} // namespace april_math
#endif

namespace april_math {
  
  void doApplySoftmaxActivation(FloatGPUMirroredMemoryBlock *input_units,
                                FloatGPUMirroredMemoryBlock *output_units,
                                FloatGPUMirroredMemoryBlock *minimums,
                                FloatGPUMirroredMemoryBlock *maximums,
                                FloatGPUMirroredMemoryBlock *sums,
                                unsigned int size,
                                unsigned int bunch_size,
                                bool use_gpu) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(minimums);
    UNUSED_VARIABLE(maximums);
    UNUSED_VARIABLE(sums);
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      const float *input_units_ptr = input_units->getGPUForRead();
      float *output_units_ptr      = output_units->getGPUForWrite();
      float *minimums_ptr          = minimums->getGPUForWrite();
      float *maximums_ptr          = maximums->getGPUForWrite();
      float *sums_ptr              = sums->getGPUForWrite();
      unsigned int units_top       = ceilingPowerOfTwo(size);
      unsigned int top_reduction   = units_top;
      dim3 block, grid;
      computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                          block, grid);
    
      minMaxFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (input_units_ptr,
         minimums_ptr,
         maximums_ptr,
         top_reduction,
         size,
         bunch_size,
         bunch_size);
      for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
        computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                            block, grid);
        minMaxNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
          (minimums_ptr,
           maximums_ptr,
           top_reduction,
           bunch_size,
           bunch_size);
      }

      computeBlockAndGridSizesForArray(bunch_size, block, grid);
      applyMinimumNorm<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (minimums_ptr,
         maximums_ptr,
         bunch_size);

      computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                          block, grid);

      applyExpMinus<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (input_units_ptr,
         output_units_ptr,
         minimums_ptr,
         size,
         bunch_size,
         bunch_size);
    
      // We reset the top
      top_reduction = units_top;

      sumFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (output_units_ptr,
         sums_ptr,
         top_reduction,
         size,
         bunch_size,
         bunch_size);
      for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
        computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                            block, grid);
        sumNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
          (sums_ptr,
           top_reduction,
           bunch_size,
           bunch_size);
      }

      computeBlockAndGridSizesForArray(bunch_size, block, grid);
      applyInverse<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (sums_ptr,
         bunch_size);

      computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                          block, grid);

      applyRatio<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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

  void doMultiplySoftmaxDerivatives(FloatGPUMirroredMemoryBlock *output_units,
                                    FloatGPUMirroredMemoryBlock *input_errors,
                                    FloatGPUMirroredMemoryBlock *output_errors,
                                    unsigned int size,
                                    unsigned int bunch_size,
                                    bool use_gpu) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
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

  void doApplyLogSoftmaxActivation(FloatGPUMirroredMemoryBlock *input_units,
                                   FloatGPUMirroredMemoryBlock *output_units,
                                   FloatGPUMirroredMemoryBlock *minimums,
                                   FloatGPUMirroredMemoryBlock *maximums,
                                   FloatGPUMirroredMemoryBlock *sums,
                                   unsigned int size,
                                   unsigned int bunch_size,
                                   bool use_gpu) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(minimums);
    UNUSED_VARIABLE(maximums);
    UNUSED_VARIABLE(sums);
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      const float *input_units_ptr = input_units->getGPUForRead();
      float *output_units_ptr      = output_units->getGPUForWrite();
      float *minimums_ptr          = minimums->getGPUForWrite();
      float *maximums_ptr          = maximums->getGPUForWrite();
      float *sums_ptr              = sums->getGPUForWrite();
      unsigned int units_top       = ceilingPowerOfTwo(size);
      unsigned int top_reduction   = units_top;
      dim3 block, grid;
      computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                          block, grid);
    
      minMaxFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (input_units_ptr,
         minimums_ptr,
         maximums_ptr,
         top_reduction,
         size,
         bunch_size,
         bunch_size);
      for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
        computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                            block, grid);
        minMaxNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
          (minimums_ptr,
           maximums_ptr,
           top_reduction,
           bunch_size,
           bunch_size);
      }
      
      computeBlockAndGridSizesForArray(bunch_size, block, grid);
      applyMinimumNorm<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (minimums_ptr,
         maximums_ptr,
         bunch_size);

      computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                          block, grid);

      applyExpMinus<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (input_units_ptr,
         output_units_ptr,
         maximums_ptr,
         size,
         bunch_size,
         bunch_size);
    
      // We reset the top
      top_reduction = units_top;

      sumFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (output_units_ptr,
         sums_ptr,
         top_reduction,
         size,
         bunch_size,
         bunch_size);
      for (top_reduction >>= 1; top_reduction > 1; top_reduction >>= 1) {
        computeBlockAndGridSizesFor2DMatrix(bunch_size, top_reduction,
                                            block, grid);
        sumNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
          (sums_ptr,
           top_reduction,
           bunch_size,
           bunch_size);
      }
      
      computeBlockAndGridSizesFor2DMatrix(bunch_size, size,
                                          block, grid);

      applyMinusLog<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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
          double exp_output = exp(static_cast<double>(output_units_ptr[cur_pos]));
          addition += exp_output;
          cur_pos  += bunch_size;
        }
        float ratio = static_cast<float>(log(addition));
        cur_pos = 0;
        for (unsigned int i = 0; i < size; i++) {
          output_units_ptr[cur_pos] -= ratio;
          assert(!(output_units_ptr[cur_pos] > 0.0f) &&
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

} // namespace april_math
