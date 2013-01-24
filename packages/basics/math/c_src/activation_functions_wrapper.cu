/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
 *
 * The APRIL-MLP toolkit is free software; you can redistribute it and/or modify it
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
#include <cmath>
#include "clamp.h"
#include "wrapper.h"
#include "ceiling_power_of_two.h"

using april_utils::clamp;
using april_utils::ceilingPowerOfTwo;

///////////////////////////////////////////////////////////
/////////////////// Kernels ///////////////////////////////
///////////////////////////////////////////////////////////
#define clip(v,min,max) ((v)<min?min:((v)>max?max:v))

#ifdef USE_CUDA
#include "cuda_utils.h"

__global__ void logisticActKernel(float *units,
                                  unsigned int max_x,
				  unsigned int lda_x,
                                  unsigned int max_y) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
    unsigned int index = getMatrixFlatIndex(matrix_x_pos, lda_x, matrix_y_pos);
    units[index] = sigmoid(1.0f, units[index]);
  }
}

__global__ void logisticDerKernel(const float *units,
                                  float *errors,
                                  unsigned int max_x,
				  unsigned int lda_x,
                                  unsigned int max_y) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
    unsigned int index = getMatrixFlatIndex(matrix_x_pos, lda_x, matrix_y_pos);
    float value = clip(units[index], 0.00001f, 0.99999f);
    errors[index] *= value * (1.0f - value);
  }
}

__global__ void tanhActKernel(float *units,
                              unsigned int max_x,
			      unsigned int lda_x,
                              unsigned int max_y) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
    unsigned int index = getMatrixFlatIndex(matrix_x_pos, lda_x, matrix_y_pos);
    units[index] = sigmoid(2.0f, units[index]) - 1.0f;
  }
}

__global__ void tanhDerKernel(const float *units,
                              float *errors,
                              unsigned int max_x,
			      unsigned int lda_x,
                              unsigned int max_y) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
    unsigned int index = getMatrixFlatIndex(matrix_x_pos, lda_x, matrix_y_pos);
    float value = clip(units[index], -0.99998f, 0.99998f);;
    errors[index] *= 0.5f * (1.0f - (value * value));
  }
}

__global__ void minMaxFirstReduction(float *units,
                                     float *minimums,
                                     float *maximums,
                                     unsigned int reduction_top,
                                     unsigned int max_x,
                                     unsigned int max_y,
                                     unsigned int lda_x) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  unsigned int active_reduction = reduction_top >> 1;

  if (matrix_y_pos < max_y) {
    if (matrix_x_pos < active_reduction) {
      unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction), lda_x, matrix_y_pos);

      if (matrix_x_pos + active_reduction < max_x) {
        if (units[index] > units[passive_index]) {
          minimums[index] = units[passive_index];
          maximums[index] = units[index];
        }
        else {
          minimums[index] = units[index];
          maximums[index] = units[passive_index];
        }
      }
      else {
        minimums[index] = units[index];
        maximums[index] = units[index];
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
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  unsigned int active_reduction = reduction_top >> 1;

  if (matrix_y_pos < max_y) {
    if (matrix_x_pos < active_reduction) {
      unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction), lda_x, matrix_y_pos);

      if (minimums[index] > minimums[passive_index])
        minimums[index] = minimums[passive_index];
      // else we do not modify anything
      if (maximums[index] < maximums[passive_index])
        maximums[index] = maximums[passive_index];
    }
  }
}

__global__ void sumFirstReduction(float *units,
                                  float *sums,
                                  unsigned int reduction_top,
                                  unsigned int max_x,
                                  unsigned int max_y,
                                  unsigned int lda_x) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  unsigned int active_reduction = reduction_top >> 1;

  if (matrix_y_pos < max_y) {
    if (matrix_x_pos < active_reduction) {
      unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction), lda_x, matrix_y_pos);

      if (matrix_x_pos + active_reduction < max_x)
        sums[index] = units[index] + units[passive_index];
      else
        sums[index] = units[index];
    }
  }
}

__global__ void sumNextReduction(float *sums,
                                 unsigned int reduction_top,
                                 unsigned int max_y,
                                 unsigned int lda_x) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  unsigned int active_reduction = reduction_top >> 1;

  if (matrix_y_pos < max_y) {
    if (matrix_x_pos < active_reduction) {
      unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
      unsigned int passive_index = getMatrixIndex((matrix_x_pos + active_reduction), lda_x, matrix_y_pos);

      sums[index] = sums[index] + sums[passive_index];
    }
  }
}

__global__ void applyMinimumNorm(float *minimums,
				 float *maximums,
				 unsigned int max_x) {
  unsigned int matrix_x_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (matrix_x_pos < max_x)
    if ((maximums[matrix_x_pos] - minimums[matrix_x_pos]) > 30.0f)
      minimums[matrix_x_pos] = maximums[matrix_x_pos] - 30.0f;
}

__global__ void applyExpMinusMinimum(float *units,
                                     float *minimums,
                                     unsigned int max_x,
                                     unsigned int max_y,
                                     unsigned int lda_x) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
    unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
    units[index] = exp(units[index] - minimums[matrix_y_pos]);   
  }
}

__global__ void applyInverse(float *sums,
                             unsigned int max_x) {
  unsigned int matrix_x_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (matrix_x_pos < max_x)
    sums[matrix_x_pos] = 1.0f/sums[matrix_x_pos];
}



__global__ void applyRatio(float *units,
                           float *ratios,
                           unsigned int max_x,
                           unsigned int max_y,
                           unsigned int lda_x) {
  unsigned int matrix_x_pos, matrix_y_pos;
  getColumnMajorBunchMatrixPositions(blockIdx,
				     blockDim,
				     threadIdx,
				     matrix_x_pos,
				     matrix_y_pos);
  if (matrix_x_pos < max_x && matrix_y_pos < max_y) {
    unsigned int index = getMatrixIndex(matrix_x_pos, lda_x, matrix_y_pos);
    units[index] = ratios[matrix_y_pos] * units[index];   
  }
}                                   
#endif

///////////////////////////////////////////////////////////
///////// Activations and derivatives wrappers ////////////
///////////////////////////////////////////////////////////

void doApplyLogisticActivation(FloatGPUMirroredMemoryBlock *units,
			       unsigned int units_size,
			       const ANNConfiguration &conf,
			       bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    float *units_ptr = units->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, units_size,
						 block, grid);
    logisticActKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       units_size);
  }
  else {
#endif
    float *units_ptr = units->getPPALForReadAndWrite();
    
    for (unsigned int i=0; i<units_size; ++i) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b)  {
        // ATTENTION: In 64-bit machines is better to use exp than expf
        units_ptr[b] = sigmoid(1.0f, units_ptr[b]);
      }
      units_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

void doMultiplyLogisticDerivatives(FloatGPUMirroredMemoryBlock *units,
				   FloatGPUMirroredMemoryBlock *input_errors,
				   unsigned int units_size,
				   const ANNConfiguration &conf,
				   bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    const float *units_ptr = units->getGPUForRead();
    float *input_errors_ptr = input_errors->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, units_size,
						 block, grid);
    logisticDerKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       input_errors_ptr,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       units_size);
  }
  else {
#endif
    const float *units_ptr  = units->getPPALForRead();
    float *input_errors_ptr = input_errors->getPPALForReadAndWrite();
    for (unsigned int i=0; i<units_size; ++i) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	float value = clamp(units_ptr[b], 0.00001f, 0.99999f);
	input_errors_ptr[b] *= value*(1.0f-value);
      }
      units_ptr        += conf.max_bunch_size;
      input_errors_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

void doApplyTanhActivation(FloatGPUMirroredMemoryBlock *units,
			   unsigned int units_size,
			   const ANNConfiguration &conf,
			   bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    float *units_ptr = units->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, units_size,
						 block, grid);
    
    tanhActKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       units_size);
  }
  else {
#endif
    float *units_ptr = units->getPPALForReadAndWrite();

    for (unsigned int i=0; i<units_size; ++i) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	units_ptr[b] = sigmoid(2.0f, units_ptr[b]) - 1.0f;
      }
      units_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

void doMultiplyTanhDerivatives(FloatGPUMirroredMemoryBlock *units,
			       FloatGPUMirroredMemoryBlock *input_errors,
			       unsigned int units_size,
			       const ANNConfiguration &conf,
			       bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    const float *units_ptr = units->getGPUForRead();
    float *input_errors_ptr = input_errors->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, units_size,
						 block, grid);
    tanhDerKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       input_errors_ptr,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       units_size);
  }
  else {
#endif
    const float *units_ptr = units->getPPALForRead();
    float *input_errors_ptr = input_errors->getPPALForReadAndWrite();

    for (unsigned int i=0; i<units_size; ++i) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	float value = clamp(units_ptr[b], -0.99998f, 0.99998f);
	input_errors_ptr[b] *= 0.5f * (1.0f-value*value);
      }
      units_ptr        += conf.max_bunch_size;
      input_errors_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

void doApplySoftmaxActivation(FloatGPUMirroredMemoryBlock *units,
			      FloatGPUMirroredMemoryBlock *minimums,
			      FloatGPUMirroredMemoryBlock *maximums,
			      FloatGPUMirroredMemoryBlock *sums,
			      unsigned int units_size,
			      const ANNConfiguration &conf,
			      bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    float *units_ptr = units->getGPUForReadAndWrite();
    float *minimums_ptr = minimums->getGPUForWrite();
    float *maximums_ptr = maximums->getGPUForWrite();
    float *sums_ptr = sums->getGPUForWrite();
    unsigned int units_top = ceilingPowerOfTwo(units_size);
    unsigned int top_reduction = units_top;
    dim3 block, grid;
    computeBlockAndGridSizesForARowMajorBunch(conf, units_size,
					      block, grid);
    
    minMaxFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       minimums_ptr,
       maximums_ptr,
       top_reduction,
       units_size,
       conf.cur_bunch_size,
       conf.max_bunch_size);
    for (top_reduction >>= 1; top_reduction != 1; top_reduction >>= 1) {
      computeBlockAndGridSizesForARowMajorBunch(conf, top_reduction,
						block, grid);
      minMaxNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (minimums_ptr,
         maximums_ptr,
         top_reduction,
         conf.cur_bunch_size,
         conf.max_bunch_size);
    }

    computeBlockAndGridSizesForAnArray(conf, block, grid);
    applyMinimumNorm<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (minimums_ptr,
       maximums_ptr,
       conf.cur_bunch_size);

    computeBlockAndGridSizesForARowMajorBunch(conf, units_size,
					      block, grid);

    applyExpMinusMinimum<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       minimums_ptr,
       units_size,
       conf.cur_bunch_size,
       conf.max_bunch_size);
    
    // We reset the top
    top_reduction = units_top;

    sumFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       sums_ptr,
       top_reduction,
       units_size,
       conf.cur_bunch_size,
       conf.max_bunch_size);
    for (top_reduction >>= 1; top_reduction != 1; top_reduction >>= 1) {
      computeBlockAndGridSizesForARowMajorBunch(conf, top_reduction,
						block, grid);
      sumNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (sums_ptr,
         top_reduction,
         conf.cur_bunch_size,
         conf.max_bunch_size);
    }

    computeBlockAndGridSizesForAnArray(conf, block, grid);
    applyInverse<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (sums_ptr,
       conf.cur_bunch_size);

    computeBlockAndGridSizesForARowMajorBunch(conf, units_size,
					      block, grid);

    applyRatio<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (units_ptr,
       sums_ptr,
       units_size,
       conf.cur_bunch_size,
       conf.max_bunch_size);
  }
  else {
#endif
    float *units_ptr = units->getPPALForReadAndWrite();
    
    for (unsigned int b = 0; b < conf.cur_bunch_size; ++b)
      {
	float minimum = units_ptr[0];
	float maximum = units_ptr[0];
	unsigned int cur_pos = conf.max_bunch_size;
	for (unsigned int i = 2; i < units_size; i += 2) {
	  float prev_unit = units_ptr[cur_pos];
	  cur_pos += conf.max_bunch_size;
	  float cur_unit = units_ptr[cur_pos];

	  if (prev_unit < cur_unit) {
	    if (prev_unit < minimum) minimum = prev_unit;
	    if (cur_unit > maximum) maximum = cur_unit;
	  } else {
	    if (cur_unit < minimum) minimum = cur_unit;
	    if (prev_unit > maximum) maximum = prev_unit;
	  }
	  cur_pos += conf.max_bunch_size;
	}
	if ((units_size & 1) == 0) { // si es par
	  unsigned int max_pos = (units_size - 1) * conf.max_bunch_size;
	  if (units_ptr[max_pos] < minimum) minimum = units_ptr[max_pos];
	  if (units_ptr[max_pos] > maximum) maximum = units_ptr[max_pos];
	}
	if ((maximum - minimum) > 30.0f) minimum = maximum - 30.0f;
	double addition = 0;
	cur_pos = 0;
	for (unsigned int i = 0; i < units_size; i++) {
	  units_ptr[cur_pos] = exp(units_ptr[cur_pos] - minimum);
	  addition += units_ptr[cur_pos];
	  cur_pos += conf.max_bunch_size;
	}
	float ratio = 1.0f/addition;
	cblas_sscal(units_size, ratio, units_ptr, conf.max_bunch_size);
	units_ptr++;
      }
#ifdef USE_CUDA
  }
#endif
}
#undef clip
