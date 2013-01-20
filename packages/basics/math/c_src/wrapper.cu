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
#include "error_print.h"
#include "wrapper.h"
#include "ceiling_power_of_two.h"

#define DERIVATIVE_SATURATION 17.0f

using april_utils::ceilingPowerOfTwo;
using april_utils::clamp;

// ATTENTION: In 64-bit machines is better to use exp than expf
#define sigmoid(numerator,value) (numerator) / (exp(-(value))+1.0f)

///////////////////////////////////////////////////////////
/////////////// Auxiliar functions ////////////////////////
///////////////////////////////////////////////////////////

#ifdef USE_CUDA
__device__ void getColumnMajorBunchMatrixPositions(const dim3 &blockIdx,
						   const dim3 &blockDim,
						   const dim3 &threadIdx,
						   unsigned int &matrix_x_pos,
						   unsigned int &matrix_y_pos) {
  matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
  matrix_y_pos = (blockIdx.y*blockDim.y + threadIdx.y);
}

#define getMatrixFlatIndex(x,lda,y) ((x)+(y)*(lda))
#define getMatrixIndex(x,lda,y) ((x)*(lda)+(y))

void computeBlockAndGridSizesForAColumnMajorBunch(const ANNConfiguration &conf,
						  unsigned int size,
						  dim3 &block, dim3 &grid) {
  const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
  // Number of threads on each block dimension
  block.x = min(MAX_THREADS, conf.cur_bunch_size);
  block.y = min(MAX_THREADS/block.x, size);
  block.z = 1;
  
  grid.x = (conf.cur_bunch_size/block.x +
	    (conf.cur_bunch_size % block.x ? 1 : 0));
  grid.y = (size/block.y + (size % block.y ? 1 : 0));
  grid.z = 1;
  // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
}

void computeBlockAndGridSizesForARowMajorBunch(const ANNConfiguration &conf,
					       unsigned int size,
					       dim3 &block, dim3 &grid) {
  const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
  // Number of threads on each block dimension
  block.x = min(MAX_THREADS, size);
  block.y = min(MAX_THREADS/block.x, conf.cur_bunch_size);
  block.z = 1;
  
  grid.x = (size/block.x +
	    (size % block.x ? 1 : 0));
  grid.y = (conf.cur_bunch_size/block.y + (conf.cur_bunch_size % block.y ? 1 : 0));
  grid.z = 1;
  // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
}

void computeBlockAndGridSizesForAnArray(const ANNConfiguration &conf,
					dim3 &block, dim3 &grid) {
  const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
  // Number of threads on each block dimension
  block.x = min(MAX_THREADS, conf.cur_bunch_size);
  block.y = 1;
  block.z = 1;
  
  grid.x = (conf.cur_bunch_size/block.x +
	    (conf.cur_bunch_size % block.x ? 1 : 0));
  grid.y = 1;
  grid.z = 1;
  // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
}

cublasOperation_t getCublasOperation(CBLAS_TRANSPOSE operation) {
  if (operation == CblasNoTrans)
    return CUBLAS_OP_N;
  else if (operation == CblasTrans)
    return CUBLAS_OP_T;
  else // operation == CblasConjTrans
    return CUBLAS_OP_C;
}
#endif

///////////////////////////////////////////////////////////
/////////////////// Kernels ///////////////////////////////
///////////////////////////////////////////////////////////

#ifdef USE_CUDA
__global__ void scopyLoopKernel(unsigned int N,
				const float *x_mem,
				unsigned int x_inc,
				float *y_mem,
				unsigned int y_inc,
				unsigned int times,
				unsigned int y_ld) {
  
  unsigned int matrix_x_pos, matrix_y_pos;
  matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
  matrix_y_pos = blockIdx.y*blockDim.y + threadIdx.y;
  if (matrix_x_pos < times && matrix_y_pos < N) {
    unsigned int index_x = matrix_y_pos*x_inc;
    unsigned int index_y = matrix_x_pos*y_ld + matrix_y_pos*y_inc;
    y_mem[index_y] = x_mem[index_x];
  }
}

__global__ void saxpyLoopKernel(unsigned int N,
				float alpha,
				const float *x_mem,
				unsigned int x_inc,
				float *y_mem,
				unsigned int y_inc,
				unsigned int times,
				unsigned int x_ld) {
  unsigned int matrix_x_pos, matrix_y_pos;
  matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
  matrix_y_pos = blockIdx.y*blockDim.y + threadIdx.y;
  if (matrix_x_pos < times && matrix_y_pos < N) {
    unsigned int index_x = matrix_x_pos*x_ld + matrix_y_pos*x_inc;
    unsigned int index_y = matrix_y_pos*y_inc;
    float val = alpha * x_mem[index_x];
    // This loop is used to synchronize the threads for accessing
    // the global memory where they write the results. The loop
    // gets all the values from the threads at the index X in 
    // the current block, synchronizing the access to Y.
    for (unsigned int i=0; i<blockDim.x; ++i) {
      if (i==threadIdx.x) y_mem[index_y] += val;
      __syncthreads();
    }
  }
}

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
    float value = clamp(units[index], 0.0000001f, 0.9999999f);
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
    float value = clamp(units[index], -0.99999998f, 0.99999998f);;
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

__global__ void applyMSEKernel(const float *output,
                               const float *target_output,
                               float *output_error,
                               float *pattern_errors,
                               float zero_epsilon_distance,
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
    float d = output_error[index] = output[index] - target_output[index];
    if (fabsf(d) < zero_epsilon_distance)
      output_error[index] = d = 0.0f;
    pattern_errors[index] += d*d;
  }
}

__global__ void applyTanhKernel(const float *output,
                                const float *target_output,
                                float *output_error,
                                float *pattern_errors,
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
    float d = output_error[index] = output[index] - target_output[index];
    if (d < -0.9999999f)
      output_error[index] = -DERIVATIVE_SATURATION;
    else if (d > 0.9999999f)
      output_error[index] =  DERIVATIVE_SATURATION;
    else output_error[index] = log((1.0f+output_error[index])/(1.0f-output_error[index]));
    pattern_errors[index] += d*d;
  }
}

__global__ void applyCrossEntropyKernel(const float *output,
                                        const float *target_output,
                                        float *output_error,
                                        float *pattern_errors,
                                        float epsilon,
                                        float inf,
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
    output_error[index] = output[index] - target_output[index];
    if (fabsf(target_output[index]) > epsilon) {
      if (fabsf(output[index] > epsilon))
	pattern_errors[index] += target_output[index]*logf(output[index]);
      else
	pattern_errors[index] += target_output[index]*inf;
    }
  }
}

__global__ void applyFullCrossEntropyKernel(const float *output,
                                            const float *target_output,
                                            float *output_error,
                                            float *pattern_errors,
                                            float epsilon,
                                            float inf,
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
    // FIXME: CUIDADO esto solo es cierto para salida LOGISTIC o SOFTMAX
    output_error[index] = output[index] - target_output[index];
    float aux_out    = output_error[index];
    float aux_target = target_output[index];
    if (fabs(aux_target) > epsilon)
      pattern_errors[index] += aux_target * ((fabs(aux_out) > epsilon) ? logf(aux_out) : inf);
    if (fabs(1.0f - aux_target) > epsilon)
      pattern_errors[index] += (1.0f - aux_target) * ((fabs(1.0f - aux_out) > epsilon) ? logf(1.0f - aux_out) : inf);
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
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) 
        // ATTENTION: In 64-bit machines is better to use exp than expf
        units_ptr[b] = sigmoid(1.0f, units_ptr[b]);
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
	float value = clamp(units_ptr[b], 0.000000001f, 0.999999999f);
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
        float value = clamp(units_ptr[b], -0.99999998f, 0.99999998f);
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

///////////////////////////////////////////////////////////
///////////////// Error functions wrappers ////////////////
///////////////////////////////////////////////////////////

void doCalculateMSE(FloatGPUMirroredMemoryBlock *output,
		    FloatGPUMirroredMemoryBlock *target_output,
		    FloatGPUMirroredMemoryBlock *output_error,
		    FloatGPUMirroredMemoryBlock *pattern_errors,
		    float zero_epsilon_distance,
		    unsigned int output_size,
		    const ANNConfiguration &conf,
		    bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {    
    const float *output_ptr        = output->getGPUForRead();
    const float *target_output_ptr = target_output->getGPUForRead();
    float *output_error_ptr        = output_error->getGPUForWrite();
    float *pattern_errors_ptr      = pattern_errors->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, output_size,
						 block, grid);
    applyMSEKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (output_ptr,
       target_output_ptr,
       output_error_ptr,
       pattern_errors_ptr,
       zero_epsilon_distance,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       output_size);
  }
  else {
#endif
    float d = 0;
    const float *output_ptr        = output->getPPALForRead();
    const float *target_output_ptr = target_output->getPPALForRead();
    float *output_error_ptr        = output_error->getPPALForWrite();
    float *pattern_errors_ptr      = pattern_errors->getPPALForReadAndWrite();
    
    for (unsigned int i = 0; i < output_size; i++) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	d = output_error_ptr[b] = output_ptr[b] - target_output_ptr[b];
	if (fabsf(d) < zero_epsilon_distance)
	  output_error_ptr[b] = d = 0.0f;
	pattern_errors_ptr[b] += d*d;
      }
      output_ptr         += conf.max_bunch_size;
      target_output_ptr  += conf.max_bunch_size;
      output_error_ptr   += conf.max_bunch_size;
      pattern_errors_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

void doCalculateTanh(FloatGPUMirroredMemoryBlock *output,
		     FloatGPUMirroredMemoryBlock *target_output,
		     FloatGPUMirroredMemoryBlock *output_error,
		     FloatGPUMirroredMemoryBlock *pattern_errors,
		     unsigned int output_size,
		     const ANNConfiguration &conf,
		     bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    const float *output_ptr        = output->getGPUForRead();
    const float *target_output_ptr = target_output->getGPUForRead();
    float *output_error_ptr        = output_error->getGPUForWrite();
    float *pattern_errors_ptr      = pattern_errors->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, output_size,
						 block, grid);
  
    applyTanhKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (output_ptr,
       target_output_ptr,
       output_error_ptr,
       pattern_errors_ptr,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       output_size);
  }
  else {
#endif
    float d = 0;
    const float *output_ptr        = output->getPPALForRead();
    const float *target_output_ptr = target_output->getPPALForRead();
    float *output_error_ptr        = output_error->getPPALForWrite();
    float *pattern_errors_ptr      = pattern_errors->getPPALForReadAndWrite();
    
    for (unsigned int i = 0; i < output_size; i++) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
        d = output_error_ptr[b] = output_ptr[b] - target_output_ptr[b];
        if (d < -0.9999999f)
          output_error_ptr[b] = -DERIVATIVE_SATURATION;
        else if (d > 0.9999999f)
          output_error_ptr[b] =  DERIVATIVE_SATURATION;
        else output_error_ptr[b] = log((1.0f+output_error_ptr[b])/(1.0f-output_error_ptr[b]));
	pattern_errors_ptr[b] += d*d;
      }
      output_ptr         += conf.max_bunch_size;
      target_output_ptr  += conf.max_bunch_size;
      output_error_ptr   += conf.max_bunch_size;
      pattern_errors_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

/*
  void doCalculateMixtureCrossEntropy(FloatGPUMirroredMemoryBlock *output,
  FloatGPUMirroredMemoryBlock *target_output,
  FloatGPUMirroredMemoryBlock *output_error,
  FloatGPUMirroredMemoryBlock *pattern_errors,
  float EPSILON,
  float INF,
  unsigned int output_size,
  const ANNConfiguration &conf,
  bool use_gpu) {
  const float *output_ptr        = output->getPPALForRead();
  const float *target_output_ptr = target_output->getPPALForRead();
  float *output_error_ptr        = output_error->getPPALForWrite();
  float *pattern_errors_ptr      = pattern_errors->getGPUForReadAndWrite();

  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
  float Z = 0.0f;
  unsigned int ipos = b;
  for (unsigned int i=0; i<output_size; ++i)
  {
  Z += target_output_ptr[ipos] * output_ptr[ipos];
  ipos += conf.max_bunch_size;
  }
  Z = 1.0f/Z;
  float prob = 0.0f;
  ipos = b;
  for (unsigned int i = 0; i < output_size; i++) {
  float component_prob = target_output_ptr[ipos] * output_ptr[ipos];
  output_error_ptr[ipos] = output_ptr[ipos] - component_prob*Z;
  prob += component_prob;
  ipos += conf.max_bunch_size;
  }
  s += ((fabs(prob) > EPSILON) ? logf(prob) : INF);
  }
  return s;
  }
*/

// F'(a,b)/a_i = ( 2 b_i H(a,b) - G(a,b) ) / H^2(a,b)
void doCalculateLocalFMeasure(float alpha,
			      FloatGPUMirroredMemoryBlock *output,
			      FloatGPUMirroredMemoryBlock *target_output,
			      FloatGPUMirroredMemoryBlock *output_error,
			      FloatGPUMirroredMemoryBlock *pattern_errors,
			      unsigned int output_size,
			      const ANNConfiguration &conf,
			      bool use_gpu) {
  if (use_gpu) ERROR_EXIT(128, "GPU VERSION NOT IMPLEMENTED!!!\n");
  const float *output_ptr        = output->getPPALForRead();
  const float *target_output_ptr = target_output->getPPALForRead();
  float *pattern_errors_ptr      = pattern_errors->getPPALForReadAndWrite();
  float *output_error_ptr        = output_error->getPPALForWrite();
  float Gab = 0.0f, Hab = 0.0f;
  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
    unsigned int ipos = b;
    for (unsigned int i = 0; i < output_size; i++) {
      // float out = clamp(output_ptr[ipos], 0.0f, 1.0f);
      float out = output_ptr[ipos];
      Gab += 1.0f + out * target_output_ptr[ipos] - out - target_output_ptr[ipos];
      Hab += 2.0f - out - target_output_ptr[ipos];
      ipos += conf.max_bunch_size;
    }
  }
  float HabP2 = Hab*Hab;
  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
    unsigned int ipos = b;
    for (unsigned int i = 0; i < output_size; i++) {
      // Aqui cambiamos de signo alpha para convertir una minimizacion en una
      // maximizacion
      if (HabP2 > 0.0f) {
	float v = -alpha * ( (target_output_ptr[ipos] - 1) * Hab + Gab) / HabP2;
	output_error_ptr[ipos] = clamp(v,
				       -DERIVATIVE_SATURATION,
				       DERIVATIVE_SATURATION);
      }
      else output_error_ptr[ipos] = 0.0f;
      ipos += conf.max_bunch_size;
    }
  }
  // cambiamos de signo para convertir la minimizacion en una maximizacion
  float error;
  if (Hab > 0.0f)
    error = -alpha*Gab/Hab;
  else error = -1.0f;
  // Sumamos el error en la componente 0 porque la FMeasure no se descompone por
  // neuronas y bunch, se calcula para todo el grupo de neuronas y para todo el
  // bunch de una sola vez
  pattern_errors_ptr[0] += error;
}

/*
  float doCalculateGA(FloatGPUMirroredMemoryBlock *output,
  FloatGPUMirroredMemoryBlock *target_output,
  FloatGPUMirroredMemoryBlock *output_error,
  FloatGPUMirroredMemoryBlock *pattern_errors,
  unsigned int output_size,
  const ANNConfiguration &conf,
  bool use_gpu) {
  const float *output_ptr        = output->getPPALForRead();
  const float *target_output_ptr = target_output->getPPALForRead();
  float *output_error_ptr        = output_error->getPPALForWrite();

  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
  // Las 2 siguientes variables no se emplean?
  //float sum_a_b = 0.0f;
  //float sum_c_a_b;
  float Gab = 0.0f, Hab = 0.0f;
  unsigned int ipos = b;
  for (unsigned int i = 0; i < output_size; i++) {
  Gab += output_ptr[ipos] * target_output_ptr[ipos];
  Hab += output_ptr[ipos] + target_output_ptr[ipos];
  ipos += conf.max_bunch_size;
  }
  Gab *= 2.0f;
  s   += 1.0f - Gab/Hab; // hacemos 1 - FMeasure para cambiar la minimizacion
  // por una maximizacion
  float HabP2 = Hab*Hab;
  ipos = b;
  for (unsigned int i = 0; i < output_size; i++) {
  // Aqui cambiamos de signo para convertir una minimizacion en una
  // maximizacion
  output_error_ptr[ipos] = -(2 * target_output_ptr[ipos] * Hab - Gab) / HabP2;
  ipos += conf.max_bunch_size;
  }
  }
  return s;
  }

*/

void doCalculateCrossEntropy(FloatGPUMirroredMemoryBlock *output,
			     FloatGPUMirroredMemoryBlock *target_output,
			     FloatGPUMirroredMemoryBlock *output_error,
			     FloatGPUMirroredMemoryBlock *pattern_errors,
			     float EPSILON,
			     float INF,
			     unsigned int output_size,
			     const ANNConfiguration &conf,
			     bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    const float *output_ptr        = output->getGPUForRead();
    const float *target_output_ptr = target_output->getGPUForRead();
    float *output_error_ptr        = output_error->getGPUForWrite();
    float *pattern_errors_ptr      = pattern_errors->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, output_size,
						 block, grid);

    applyCrossEntropyKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (output_ptr,
       target_output_ptr,
       output_error_ptr,
       pattern_errors_ptr,
       EPSILON,
       INF,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       output_size);
  }
  else {
#endif
    const float *output_ptr        = output->getPPALForRead();
    const float *target_output_ptr = target_output->getPPALForRead();
    float *output_error_ptr        = output_error->getPPALForWrite();
    float *pattern_errors_ptr      = pattern_errors->getPPALForReadAndWrite();

    for (unsigned int i = 0; i < output_size; i++) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	output_error_ptr[b] = output_ptr[b] - target_output_ptr[b];
	if (fabs(target_output_ptr[b]) > EPSILON)
	  pattern_errors_ptr[b] += target_output_ptr[b] * ((fabs(output_ptr[b]) > EPSILON) ?
							   logf(output_ptr[b]) : INF);
      }
      output_ptr         += conf.max_bunch_size;
      target_output_ptr  += conf.max_bunch_size;
      output_error_ptr   += conf.max_bunch_size;
      pattern_errors_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}

void doCalculateFullCrossEntropy(FloatGPUMirroredMemoryBlock *output,
				 FloatGPUMirroredMemoryBlock *target_output,
				 FloatGPUMirroredMemoryBlock *output_error,
				 FloatGPUMirroredMemoryBlock *pattern_errors,
				 float EPSILON,
				 float INF,
				 unsigned int output_size,
				 const ANNConfiguration &conf,
				 bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    const float *output_ptr        = output->getGPUForRead();
    const float *target_output_ptr = target_output->getGPUForRead();
    float *output_error_ptr        = output_error->getGPUForWrite();
    float *pattern_errors_ptr      = pattern_errors->getGPUForReadAndWrite();
    dim3 block, grid;
    computeBlockAndGridSizesForAColumnMajorBunch(conf, output_size,
						 block, grid);

    applyFullCrossEntropyKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (output_ptr,
       target_output_ptr,
       output_error_ptr,
       pattern_errors_ptr,
       EPSILON,
       INF,
       conf.cur_bunch_size,
       conf.max_bunch_size,
       output_size);

  }
  else {
#endif
    const float *output_ptr        = output->getPPALForRead();
    const float *target_output_ptr = target_output->getPPALForRead();
    float *output_error_ptr        = output_error->getPPALForWrite();
    float *pattern_errors_ptr      = pattern_errors->getPPALForReadAndWrite();

    for (unsigned int i = 0; i < output_size; i++) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	output_error_ptr[b] = output_ptr[b] - target_output_ptr[b];
	float aux_out    = output_error_ptr[b];
	float aux_target = target_output_ptr[b];
	if (fabs(aux_target) > EPSILON) {
	  pattern_errors_ptr[b] += aux_target * ((fabs(aux_out) > EPSILON) ?
						 logf(aux_out) : INF);
	}
	if (fabs(1.0f - aux_target) > EPSILON) {
	  pattern_errors_ptr[b] += (1.0f - aux_target) * ((fabs(1.0f - aux_out) > EPSILON) ?
							  logf(1.0f - aux_out) : INF);
	}
      }
      output_ptr         += conf.max_bunch_size;
      target_output_ptr  += conf.max_bunch_size;
      output_error_ptr   += conf.max_bunch_size;
      pattern_errors_ptr += conf.max_bunch_size;
    }
#ifdef USE_CUDA
  }
#endif
}


///////////////////////////////////////////////////////////
//////////////////// BLAS wrappers ////////////////////////
///////////////////////////////////////////////////////////

void doSgemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	     int m, int n,
	     float alpha, FloatGPUMirroredMemoryBlock *a, unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock *x, unsigned int x_inc,
	     float beta, FloatGPUMirroredMemoryBlock *y, unsigned int y_inc,
	     unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	     bool use_gpu) {
  const float *a_mem, *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    assert(major_type == CblasColMajor);
    cublasOperation_t cublas_a_transpose = getCublasOperation(a_transpose);
    a_mem = a->getGPUForRead() + a_shift;
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSgemv(handle, cublas_a_transpose,
			 m, n,
			 &alpha, a_mem, a_inc,
			 x_mem, x_inc,
			 &beta, y_mem, y_inc);

    checkCublasError(status);
  }
  else {
#endif
    a_mem = a->getPPALForRead() + a_shift;
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;
    cblas_sgemv(major_type, a_transpose,
                m, n,
                alpha, a_mem, a_inc,
                x_mem, x_inc,
                beta, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}


void doScopy(int N, FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift,
	     unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y,
	     unsigned int y_shift,
	     unsigned int y_inc,
	     bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu)
    {
      cublasStatus_t status;
      cublasHandle_t handle = GPUHelper::getHandler();
      //printf("Doing a scopy with comp=1 & cuda=1\n");
      x_mem = x->getGPUForRead() + x_shift;
      y_mem = y->getGPUForWrite() + y_shift;

      status = cublasSetStream(handle, GPUHelper::getCurrentStream());
      checkCublasError(status);

      status = cublasScopy(handle, N, x_mem, x_inc, y_mem, y_inc);

      checkCublasError(status);
    }
  else
    {
      //printf("Doing a scopy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
      //printf("Doing a scopy with comp=0 & cuda=0\n");
#endif
      x_mem = x->getPPALForRead() + x_shift;
      y_mem = y->getPPALForWrite() + y_shift;

      cblas_scopy(N, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
    }
#endif
}

void doScopyLoop(int N,
		 FloatGPUMirroredMemoryBlock* x,
		 unsigned int x_inc,
		 FloatGPUMirroredMemoryBlock* y,
		 unsigned int y_inc,
		 unsigned int times,
		 const unsigned int stride,
		 bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu)
    {
      //printf("Doing a scopy with comp=1 & cuda=1\n");
      x_mem = x->getGPUForRead();
      y_mem = y->getGPUForWrite();

      const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
      dim3 block, grid;
      // Number of threads on each block dimension
      block.x = min(MAX_THREADS, times);
      block.y = min(MAX_THREADS/block.x, N);
      block.z = 1;

      grid.x = (times/block.x +
                (times % block.x ? 1 : 0));
      grid.y = (N/block.y + (N % block.y ? 1 : 0));
      grid.z = 1;

      scopyLoopKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
	(N, x_mem, x_inc, y_mem, y_inc, times, stride);
    }
  else
    {
      //printf("Doing a scopy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
      //printf("Doing a scopy with comp=0 & cuda=0\n");
#endif
      x_mem = x->getPPALForRead();
      y_mem = y->getPPALForWrite();

      for (unsigned int i = 0; i < times; i++)
	cblas_scopy(N, 
                    x_mem, x_inc,
                    y_mem + i * stride , y_inc);
#ifdef USE_CUDA
    }
#endif
}

void doSaxpy(int N,
	     float alpha,
	     FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift,
	     unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y,
	     unsigned int y_shift,
	     unsigned int y_inc,
	     bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu)
    {
      cublasStatus_t status;
      cublasHandle_t handle = GPUHelper::getHandler();
      //printf("Doing a saxpy with comp=1 & cuda=1\n");
      x_mem = x->getGPUForRead() + x_shift;
      y_mem = y->getGPUForReadAndWrite() + y_shift;

      status = cublasSetStream(handle, GPUHelper::getCurrentStream());
      checkCublasError(status);

      status = cublasSaxpy(handle, N, &alpha, x_mem, x_inc, y_mem, y_inc);

      checkCublasError(status);
    }
  else
    {
      //printf("Doing a saxpy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
      //printf("Doing a saxpy with comp=0 & cuda=0\n");
#endif
      x_mem = x->getPPALForRead() + x_shift;
      y_mem = y->getPPALForReadAndWrite() + y_shift;

      cblas_saxpy(N, alpha, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
    }
#endif
}

void doSaxpyLoop(int N,
		 float alpha,
		 FloatGPUMirroredMemoryBlock* x,
		 unsigned int x_inc,
		 FloatGPUMirroredMemoryBlock* y,
		 unsigned int y_inc,
		 unsigned int times,
		 const unsigned int stride,
		 bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu)
    {
      /*
	cublasStatus_t status;
	cublasHandle_t handle = GPUHelper::getHandler();
        //printf("Doing a saxpy loop with comp=1 & cuda=1\n");
        x_mem = x->getGPUForRead();
        y_mem = y->getGPUForReadAndWrite();

        status = cublasSetStream(handle, GPUHelper::getCurrentStream());
        checkCublasError(status);

        for (unsigned int i = 0; i < times; i++) {
        status = cublasSaxpy(handle, N, &alpha,
        x_mem + i * stride, x_inc, 
        y_mem, y_inc);

        checkCublasError(status);
        }
      */
      x_mem = x->getGPUForRead();
      y_mem = y->getGPUForWrite();

      const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
      dim3 block, grid;
      // Number of threads on each block dimension
      block.x = min(MAX_THREADS, times);
      block.y = min(MAX_THREADS/block.x, N);
      block.z = 1;

      grid.x = (times/block.x +
                (times % block.x ? 1 : 0));
      grid.y = (N/block.y + (N % block.y ? 1 : 0));
      grid.z = 1;

      saxpyLoopKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
	(N, alpha, x_mem, x_inc, y_mem, y_inc, times, stride);
    }
  else
    {
      //printf("Doing a saxpy loop with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
      //printf("Doing a saxpy loop with comp=0 & cuda=0\n");
#endif
      x_mem = x->getPPALForRead();
      y_mem = y->getPPALForReadAndWrite();

      for (unsigned int i = 0; i < times; i++)
	cblas_saxpy(N, alpha,
                    x_mem + i * stride, x_inc, 
                    y_mem, y_inc);
#ifdef USE_CUDA
    }
#endif
}

void doSgemm(CBLAS_ORDER major_type,
	     CBLAS_TRANSPOSE a_transpose,
	     CBLAS_TRANSPOSE b_transpose,
	     int m,
	     int n,
	     int k,
	     float alpha,
	     FloatGPUMirroredMemoryBlock* a,
	     unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock* b,
	     unsigned int b_inc,
	     float beta,
	     FloatGPUMirroredMemoryBlock* c,
	     unsigned int c_inc,
	     unsigned int a_shift,
	     unsigned int b_shift,
	     unsigned int c_shift,
	     bool use_gpu)
{
  const float *a_mem, *b_mem;
  float *c_mem;
#ifdef USE_CUDA
  if (use_gpu)
    {
      cublasStatus_t status;
      cublasHandle_t handle = GPUHelper::getHandler();
      assert(major_type == CblasColMajor);
      //printf("Doing a sgemm with comp=1 & cuda=1\n");
      a_mem = a->getGPUForRead() + a_shift;
      b_mem = b->getGPUForRead() + b_shift;
      c_mem = c->getGPUForReadAndWrite() + c_shift;
      cublasOperation_t cublas_a_transpose = getCublasOperation(a_transpose);
      cublasOperation_t cublas_b_transpose = getCublasOperation(b_transpose);

      status = cublasSetStream(handle, GPUHelper::getCurrentStream());
      checkCublasError(status);

      status = cublasSgemm(handle, cublas_a_transpose, cublas_b_transpose,
			   m, n, k,
			   &alpha, a_mem, a_inc,
			   b_mem, b_inc,
			   &beta, c_mem, c_inc);

      checkCublasError(status);
    }
  else
    {
      //printf("Doing a sgemm with comp=1 & cuda=0\n");
#endif
      //printf("Doing a sgemm with comp=0 & cuda=0\n");
      a_mem = a->getPPALForRead() + a_shift;
      b_mem = b->getPPALForRead() + b_shift;
      c_mem = c->getPPALForReadAndWrite() + c_shift;

      // matrix matrix product: C = \alpha op(A) op(B) + \beta C
      cblas_sgemm(major_type,   // Row or Col Major
		  a_transpose,  // Transpose or not A
		  b_transpose,  // Transpose or not B
		  m,            // num rows of A (before transpose)
		  n,            // num rows at B (before transpose)
		  k,            // Common dimension between A and B
		  alpha,        // Alpha value
		  a_mem,        // A matrix
		  a_inc,        // A matrix stride
		  b_mem,        // B matrix
		  b_inc,        // B matrix stride
		  beta,         // Beta value
		  c_mem,        // C matrix
		  c_inc);       // C matrix stride
#ifdef USE_CUDA
    }
#endif
}

void doVectorSetToZero(FloatGPUMirroredMemoryBlock *v,
		       unsigned int v_size,
		       unsigned int inc,
		       unsigned int shift,
		       bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    float *ptr   = v->getGPUForWrite() + shift;
    float  value = 0.0f;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSscal(handle, v_size, &value, ptr, inc);
    // FIXME: To use cuMemsetD32 instead of cublasSscal
    checkCublasError(status);
  }
  else {
#endif
    float *ptr = v->getPPALForWrite() + shift;
    VECTOR_SSET(v_size, 0.0f, ptr, inc);
#ifdef USE_CUDA
  }
#endif
}

void doVectorSet(FloatGPUMirroredMemoryBlock *v,
		 float value,
		 unsigned int v_size,
		 unsigned int inc,
		 unsigned int shift,
		 bool use_gpu) {
  float *ptr = v->getPPALForWrite() + shift;
  VECTOR_SSET(v_size, value, ptr, inc);
}

void doSscal(unsigned int size,
	     float alpha,
	     FloatGPUMirroredMemoryBlock *x,
	     unsigned int shift,
	     unsigned int inc,
	     bool use_gpu) {
  float *x_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    x_mem = x->getGPUForReadAndWrite() + shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSscal(handle, size, &alpha, x_mem, inc);

    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForReadAndWrite() + shift;
    cblas_sscal(size, alpha, x_mem, inc);
#ifdef USE_CUDA
  }
#endif
}

void doSger(CBLAS_ORDER major_type,
	    unsigned int m,
	    unsigned int n,
	    float alpha,
	    FloatGPUMirroredMemoryBlock *x,
	    unsigned int x_shift,
	    unsigned int x_inc,
	    FloatGPUMirroredMemoryBlock *y,
	    unsigned int y_shift,
	    unsigned int y_inc,
	    FloatGPUMirroredMemoryBlock *a,
	    unsigned int a_shift,
	    unsigned int a_inc,
	    bool use_gpu) {
  const float *x_mem;
  const float *y_mem;
  float *a_mem;
#ifdef USE_CUDA

  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForRead() + y_shift;
    a_mem = a->getGPUForReadAndWrite() + a_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSger(handle,
			m, n,
			&alpha,
			x_mem, x_inc,
			y_mem, y_inc,
			a_mem, a_inc);

    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForRead() + y_shift;
    a_mem = a->getPPALForReadAndWrite() + a_shift;

    cblas_sger(major_type,
	       m, n,
	       alpha,
	       x_mem, x_inc,
	       y_mem, y_inc,
	       a_mem, a_inc);
#ifdef USE_CUDA
  }
#endif
}

#ifdef USE_CUDA
#undef getMatrixFlatIndex
#endif

#undef sigmoid
