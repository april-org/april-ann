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

using april_utils::clamp;

// ATTENTION: In 64-bit machines is better to use exp than expf
#define sigmoid(numerator,value) (numerator) / (exp(-(value))+1.0f)

///////////////////////////////////////////////////////////
/////////////////// Kernels ///////////////////////////////
///////////////////////////////////////////////////////////

#ifdef USE_CUDA
#include "cuda_utils.h"
__global__ void applyMSEErrorFunctionKernel(const float *output,
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

__global__ void applyTanhErrorFunctionKernel(const float *output,
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

__global__ void applyCrossEntropyErrorFunctionKernel(const float *output,
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
    float o = clip(output[index], epsilon, 1.0f - epsilon);
    float t = clip(target_output[index], epsilon, 1.0f - epsilon);
    if (t > epsilon) {
      if (o > epsilon) pattern_errors[index] += t * logf(o);
      else pattern_errors[index] += t * inf;
    }
    // compute derivative
    output_error[index] = (o - t) / (o * (1.0f - o));
  }
}

__global__ void applyFullCrossEntropyErrorFunctionKernel(const float *output,
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
    float o         = clip(output[index], epsilon, 1.0f - epsilon);
    float t         = clip(target_output[index], epsilon, 1.0f - epsilon);
    float inv_t     = clip(1.0f - target_output[index], epsilon, 1.0f - epsilon);
    float log_o     = (o > epsilon) ? logf(o) : inf;
    float log_inv_o = (1.0f - o > epsilon) ? logf(1.0f - o) : inf;
    if (t > epsilon)
      pattern_errors[index] += t * log_o;
    if (inv_t > epsilon)
      pattern_errors[index] += inv_t * log_inv_o;
    // compute derivative
    output_error[index] = (o - t) / (o * (1.0f - o));
  }
}
#endif


///////////////////////////////////////////////////////////
///////////////// Error functions wrappers ////////////////
///////////////////////////////////////////////////////////

void doCalculateMSEErrorFunction(FloatGPUMirroredMemoryBlock *output,
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
    applyMSEErrorFunctionKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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

void doCalculateTanhErrorFunction(FloatGPUMirroredMemoryBlock *output,
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
  
    applyTanhErrorFunctionKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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
void doCalculateLocalFMeasureErrorFunction(float alpha,
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

void doCalculateCrossEntropyErrorFunction(FloatGPUMirroredMemoryBlock *output,
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

    applyCrossEntropyErrorFunctionKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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
	assert(!(output_ptr[b] > 1.0f) && !(output_ptr[b] < 0.0f));
	assert(!(target_output_ptr[b] > 1.0f) && !(target_output_ptr[b] < 0.0f));
	float o = clamp(output_ptr[b], EPSILON, 1.0f - EPSILON);
	float t = clamp(target_output_ptr[b], EPSILON, 1.0f - EPSILON);
	if (t > EPSILON) {
	  if (o > EPSILON) pattern_errors_ptr[b] += t * logf(o);
	  else pattern_errors_ptr[b] += t * INF;
	}
	// compute derivative
	output_error_ptr[b] = (o - t) / (o * (1.0f - o));
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

void doCalculateFullCrossEntropyErrorFunction(FloatGPUMirroredMemoryBlock *output,
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

    applyFullCrossEntropyErrorFunctionKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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
	assert(!(output_ptr[b] > 1.0f) && !(output_ptr[b] < 0.0f));
	assert(!(target_output_ptr[b] > 1.0f) && !(target_output_ptr[b] < 0.0f));
	float o         = clamp(output_ptr[b], EPSILON, 1.0f - EPSILON);
	float t         = clamp(target_output_ptr[b], EPSILON, 1.0f - EPSILON);
	float inv_t     = clamp(1.0f - target_output_ptr[b], EPSILON, 1.0f - EPSILON);
	float log_o     = (o > EPSILON) ? logf(o) : INF;
	float log_inv_o = (1.0f - o > EPSILON) ? logf(1.0f - o) : INF;
	if (t > EPSILON)
	  pattern_errors_ptr[b] += t * log_o;
	if (inv_t > EPSILON)
	  pattern_errors_ptr[b] += inv_t * log_inv_o;
	// compute derivative
	output_error_ptr[b] = (o - t) / (o * (1.0f - o));
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

#undef sigmoid
