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
#include <cstdio>
#include <cmath>
#include "errorfunc.h"
#include "wrapper.h"

namespace ANN {
  
  MSE::MSE() : zero_epsilon_distance(0.0f) {
  }
  
  // calcula la funcion de error MSE para una muestra, y deja el error
  // producido para poder retropropagarlo en caso de ser necesario
  void MSE::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
					FloatGPUMirroredMemoryBlock *target_output,
					FloatGPUMirroredMemoryBlock *output_error,
					FloatGPUMirroredMemoryBlock *pattern_errors,
					unsigned int output_size,
					const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    doCalculateMSE(output,
		   target_output,
		   output_error,
		   pattern_errors,
		   zero_epsilon_distance,
		   output_size,
		   conf,
		   use_cuda_flag);
  }
  
  // calcula el MSE de un batch a partir de la suma de todas las
  // muestras y el numero de muestras
  float MSE::computeBatchErrorFunction(float error_sums,
				       unsigned int num_patterns) {
    return 0.5 * error_sums / num_patterns;
  }

  ///////////////////////////////////////////////////////////

  Tanh::Tanh() {
  }
  
  // calcula la funcion de error Tanh para una muestra, y deja el error
  // producido para poder retropropagarlo en caso de ser necesario
  void Tanh::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
					 FloatGPUMirroredMemoryBlock *target_output,
					 FloatGPUMirroredMemoryBlock *output_error,
					 FloatGPUMirroredMemoryBlock *pattern_errors,
					 unsigned int output_size,
					 const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    doCalculateTanh(output,
		    target_output,
		    output_error,
		    pattern_errors,
		    output_size,
		    conf,
		    use_cuda_flag);
  }
  
  // calcula el Tanh de un batch a partir de la suma de todas las
  // muestras y el numero de muestras
  float Tanh::computeBatchErrorFunction(float error_sums,
					unsigned int num_patterns) {
    return 0.5 * error_sums / num_patterns;
  }

  ///////////////////////////////////////////////////////////

  /*

    MixtureCrossEntropy::MixtureCrossEntropy() :
    EPSILON(NEAR_ZERO), INF(logf(NEAR_ZERO)) {
    }

    // lo mismo que para MSE pero para la crossentropy
    float MixtureCrossEntropy::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
    FloatGPUMirroredMemoryBlock *target_output,
    FloatGPUMirroredMemoryBlock *output_error,
    FloatGPUMirroredMemoryBlock *pattern_errors,
    unsigned int output_size,
    const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    float bunch_error = doCalculateMixtureCrossEntropy(output,
    target_output,
    output_error,
    pattern_errors,
    EPSILON,
    INF,
    output_size,
    conf,
    use_cuda_flag);
    return bunch_error;
    }

    // idem
    float MixtureCrossEntropy::computeBatchErrorFunction(float error_sums,
    unsigned int num_patterns) {
    return -error_sums/num_patterns;
    }

    ///////////////////////////////////////////////////////////
    */
  
  LocalFMeasure::LocalFMeasure(float alpha) : alpha(alpha), N(0) {
  }
  
  // F(a,b) = \alpha G(a,b)/H(a,b)
  // G(a,b) = \sum (1 - a - b + ab)
  // H(a,b) = \sum (2 - a - b)
  //
  // F'(a,b)/a_i = \alpha ( (-1 + b)*H(a,b) + G(a,b) ) / H^2(a,b)
  //
  // Hay que cambiar el signo de la derivada para que vaya en sentido contrario.
  // El algoritmo de descenso por gradiente minimiza la funcion objetivo, y en
  // este caso lo que queremos es maximizar.
  void LocalFMeasure::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
						  FloatGPUMirroredMemoryBlock *target_output,
						  FloatGPUMirroredMemoryBlock *output_error,
						  FloatGPUMirroredMemoryBlock *pattern_errors,
						  unsigned int output_size,
						  const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    doCalculateLocalFMeasure(alpha,
			     output,
			     target_output,
			     output_error,
			     pattern_errors,
			     output_size,
			     conf,
			     use_cuda_flag);
    ++N;
  }
  
  // idem
  float LocalFMeasure::computeBatchErrorFunction(float error_sums,
						 unsigned int num_patterns) {
    // la FMeasure lleva su propia cuenta para calcular la media sobre el numero
    // de conjuntos BUNCH, no sobre el numero de patrones
    float ret = error_sums/N;
    N = 0;
    return ret;
  }

  /*
    ////////////////////////////////////////

    GA::GA() {
    }
  
    // GA2 = ( \sum (1-a_i) (1-b_i) ) ( \sum a_i b_i)
    // GA2/a_i = (b_i - 1)( \sum a_i b_i ) + ( \sum (1 - a_i)(1 - b_i)) b_i
    //
    // Hay que cambiar el signo de la derivada para que vaya en sentido contrario.
    // El algoritmo de descenso por gradiente minimiza la funcion objetivo, y en
    // este caso lo que queremos es maximizar.
    float GA::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
    FloatGPUMirroredMemoryBlock *target_output,
    FloatGPUMirroredMemoryBlock *output_error,
    FloatGPUMirroredMemoryBlock *pattern_errors,
    unsigned int output_size,
    const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    float bunch_error = doCalculateGA(output,
    target_output,
    output_error,
    pattern_errors,
    output_size,
    conf,
    use_cuda_flag);
    return bunch_error;
    }

    // idem
    float GA::computeBatchErrorFunction(float error_sums,
    unsigned int num_patterns) {
    return error_sums/num_patterns;
    }
  */
  ////////////////////////////////////////

  CrossEntropy::CrossEntropy() :
    EPSILON(NEAR_ZERO), INF(logf(NEAR_ZERO)) {
  }

  // lo mismo que para MSE pero para la crossentropy
  void CrossEntropy::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
						 FloatGPUMirroredMemoryBlock *target_output,
						 FloatGPUMirroredMemoryBlock *output_error,
						 FloatGPUMirroredMemoryBlock *pattern_errors,
						 unsigned int output_size,
						 const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    doCalculateCrossEntropy(output,
			    target_output,
			    output_error,
			    pattern_errors,
			    EPSILON,
			    INF,
			    output_size,
			    conf,
			    use_cuda_flag);

  }

  // idem
  float CrossEntropy::computeBatchErrorFunction(float error_sums,
						unsigned int num_patterns) {
    return -error_sums/num_patterns;
  }


  ///////////////////////////////////////////////////////////

  FullCrossEntropy::FullCrossEntropy() :
    EPSILON(NEAR_ZERO), INF(logf(NEAR_ZERO)) {
  }

  // lo mismo que para MSE pero para la crossentropy
  void FullCrossEntropy::computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
						     FloatGPUMirroredMemoryBlock *target_output,
						     FloatGPUMirroredMemoryBlock *output_error,
						     FloatGPUMirroredMemoryBlock *pattern_errors,
						     unsigned int output_size,
						     const ANNConfiguration &conf) {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.cur_bunch_size > 1);
    doCalculateFullCrossEntropy(output,
				target_output,
				output_error,
				pattern_errors,
				EPSILON,
				INF,
				output_size,
				conf,
				use_cuda_flag);

  }

  // idem
  float FullCrossEntropy::computeBatchErrorFunction(float error_sums,
						    unsigned int num_patterns) {
    return -error_sums/num_patterns;
  }

  ////////////////////////////////////////
  
  NormalizedRootMSE::~NormalizedRootMSE() {
  }
  
  float NormalizedRootMSE::computeErrorFromTimeSerie(float *output_ptr,
						     float *target_output_ptr,
						     unsigned int output_size) {
    float target_mean = 0.0f;
    float numerator = 0.0f, denominator = 0.0f;
    for (unsigned int i=0; i<output_size; i++)
      target_mean += target_output_ptr[i];
    target_mean /= output_size;
    for (unsigned int i=0; i<output_size; ++i) {
      float err = (target_output_ptr[i] - output_ptr[i]);
      numerator += err*err;
      float rel = (target_output_ptr[i] - target_mean);
      denominator += rel*rel;
    }
    return sqrtf(numerator/denominator);
  }

  ////////////////////////////////////////

  RootMSE::~RootMSE() {
  }
  
  float RootMSE::computeErrorFromTimeSerie(float *output_ptr,
					   float *target_output_ptr,
					   unsigned int output_size) {
    float numerator = 0.0f;

    for (unsigned int i=0; i<output_size; ++i) {
      float err = (target_output_ptr[i] - output_ptr[i]);
      numerator += err*err;
    }
    return sqrtf(numerator/output_size);
  }

  ////////////////////////////////////////
  
  AbsoluteError::~AbsoluteError() {
  }
  
  float AbsoluteError::computeErrorFromTimeSerie(float *output_ptr,
						 float *target_output_ptr,
						 unsigned int output_size) {
    float error = 0.0f;
    for (unsigned int i=0; i<output_size; ++i)
      error += fabsf(target_output_ptr[i] - output_ptr[i]);
    return error/output_size;
  }
}
