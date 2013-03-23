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
#include <cmath>
#include <cstring>
#include "error_print.h"
#include "cblas_headers.h"
#include "activation_function.h"
#include "ceiling_power_of_two.h"

// OJO: en maquinas de 64 bits es mejor hacer exp que expf
#define sigmoid(numerator,value) (numerator) / (exp(-(value))+1.0f)

using april_utils::ceilingPowerOfTwo;

namespace ANN {

  void LogisticActivationFunction::applyActivation(FloatGPUMirroredMemoryBlock *units,
						   unsigned int units_size,
						   const ANNConfiguration &conf,
						   bool use_cuda) {
    doApplyLogisticActivation(units,
                              units_size,
                              conf,
                              use_cuda);
  }

  void LogisticActivationFunction::multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
						       FloatGPUMirroredMemoryBlock *input_errors,
						       unsigned int size,
						       const ANNConfiguration &conf,
						       bool use_cuda,
						       bool is_output) {
    if (!is_output || !conf.error_function_logistic_mandatory)
      doMultiplyLogisticDerivatives(units,
				    input_errors,
				    size,
				    conf,
				    use_cuda);
    // else nothing because the error function is prepared to logistic
    // derivatives
  }

  ActivationFunction *LogisticActivationFunction::clone() {
    return new LogisticActivationFunction();
  }

  //////////////////////////////////////////////////////////////////////////////

  void TanhActivationFunction::applyActivation(FloatGPUMirroredMemoryBlock *units,
					       unsigned int units_size,
					       const ANNConfiguration &conf,
					       bool use_cuda) {
    doApplyTanhActivation(units,
                          units_size,
                          conf,
                          use_cuda);
  }

  void TanhActivationFunction::multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
						   FloatGPUMirroredMemoryBlock *input_errors,
						   unsigned int size,
						   const ANNConfiguration &conf,
						   bool use_cuda,
						   bool is_output) {
    if (is_output && conf.error_function_logistic_mandatory)
      ERROR_EXIT(123, "The logistic or softmax activation function is"
		 " mandataroy due to the error function");
    doMultiplyTanhDerivatives(units,
                              input_errors,
                              size,
                              conf,
                              use_cuda);
  }

  ActivationFunction *TanhActivationFunction::clone() {
    return new TanhActivationFunction();
  }
  
  //////////////////////////////////////////////////////////////////////////////

  SoftmaxActivationFunction::SoftmaxActivationFunction() :
    size(0), minimums(0), maximums(0), sums(0) {
  }
  
  SoftmaxActivationFunction::~SoftmaxActivationFunction() {
    delete minimums;
    delete maximums;
    delete sums;
  }

  void SoftmaxActivationFunction::applyActivation(FloatGPUMirroredMemoryBlock *units,
						  unsigned int units_size,
						  const ANNConfiguration &conf,
						  bool use_cuda) {
    if (use_cuda) {
      if (size == 0) {
	size = units_size;
	unsigned int reduction_size = ceilingPowerOfTwo(units_size) >> 1;
	minimums = new FloatGPUMirroredMemoryBlock(reduction_size * conf.max_bunch_size);
	maximums = new FloatGPUMirroredMemoryBlock(reduction_size * conf.max_bunch_size);
	sums     = new FloatGPUMirroredMemoryBlock(reduction_size * conf.max_bunch_size);
      }
      else if (size != units_size) ERROR_EXIT(128,
					      "A softmax activation function "
					      "with use_cuda=true only could be "
					      "used in one activation_units");
    }
    doApplySoftmaxActivation(units,
                             minimums,
                             maximums,
                             sums,
                             units_size,
                             conf,
                             use_cuda);
  }
  
  void SoftmaxActivationFunction::multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
						      FloatGPUMirroredMemoryBlock *input_errors,
						      unsigned int size,
						      const ANNConfiguration &conf,
						      bool use_cuda,
						      bool is_output) {
    if (!is_output || !conf.error_function_logistic_mandatory)
      // Is the same as sigmoid
      doMultiplyLogisticDerivatives(units,
				    input_errors,
				    size,
				    conf,
				    use_cuda);
    // else nothing because the error function is prepared to logistic
    // derivatives
  }
  
  ActivationFunction *SoftmaxActivationFunction::clone() {
    return new SoftmaxActivationFunction();
  }
  
  LinearActivationFunction::LinearActivationFunction()  { }
  LinearActivationFunction::~LinearActivationFunction() { }
  void LinearActivationFunction::applyActivation(FloatGPUMirroredMemoryBlock *units, 
						 unsigned int units_size,
						 const ANNConfiguration &conf,
						 bool use_cuda) { }
  void LinearActivationFunction::multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
						     FloatGPUMirroredMemoryBlock *input_errors,
						     unsigned int size,
						     const ANNConfiguration &conf,
						     bool use_cuda,
						     bool is_output) {
    if (is_output && conf.error_function_logistic_mandatory)
      ERROR_EXIT(123, "The logistic or softmax activation function is"
		 " mandataroy due to the error function");
  }

  ActivationFunction *LinearActivationFunction::clone() {
    return new LinearActivationFunction();
  }
  
  //////////////////////////////////////////////////////////

  SoftsignActivationFunction::SoftsignActivationFunction()  { }
  SoftsignActivationFunction::~SoftsignActivationFunction() { }
  void SoftsignActivationFunction::applyActivation(FloatGPUMirroredMemoryBlock *units, 
						   unsigned int units_size,
						   const ANNConfiguration &conf,
						   bool use_cuda) {
    doApplySoftsignActivation(units,
			      units_size,
			      conf,
			      use_cuda);
  }
  void SoftsignActivationFunction::multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
						     FloatGPUMirroredMemoryBlock *input_errors,
						     unsigned int size,
						     const ANNConfiguration &conf,
						     bool use_cuda,
						     bool is_output) {
    if (is_output && conf.error_function_logistic_mandatory)
      ERROR_EXIT(123, "The logistic or softmax activation function is"
		 " mandataroy due to the error function");
    doMultiplySoftsignDerivatives(units,
				  input_errors,
				  size,
				  conf,
				  use_cuda);
  }

  ActivationFunction *SoftsignActivationFunction::clone() {
    return new SoftsignActivationFunction();
  }
  
  //////////////////////////////////////////////////////////
  
  BinarySamplingActivationFunction::
  BinarySamplingActivationFunction(MTRand *the_rand) :
    StochasticActivationFunction(the_rand) {
  }
  
  BinarySamplingActivationFunction::~BinarySamplingActivationFunction() {
  }
  
  // TODO: Actualizar la funcion para trabajar con ColumnMajor.
  void BinarySamplingActivationFunction::
  applyActivation(FloatGPUMirroredMemoryBlock *units,
		  unsigned int units_size,
		  const ANNConfiguration &conf,
		  bool use_cuda) {
    
    if (use_cuda) {
      ERROR_EXIT(128, "NOT IMPLEMENTED YET FOR USE_CUDA=TRUE\n");
    }

    float *units_ptr = units->getPPALForReadAndWrite();
    
    for (unsigned int i=0; i<units_size; ++i) {
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b)
	units_ptr[i] = sampleOne(sigmoid(1.0f, units_ptr[i]));
      units_ptr += conf.max_bunch_size;
    }
  }
  
  ActivationFunction *BinarySamplingActivationFunction::clone() {
    MTRand *new_rand = new MTRand(*the_rand);
    IncRef(new_rand);
    return new BinarySamplingActivationFunction(new_rand);
  }
  
  void BinarySamplingActivationFunction::randomize(FloatGPUMirroredMemoryBlock *units,
						   unsigned int size,
						   const ANNConfiguration &conf,
						   bool use_cuda) {
    
    float *units_ptr = units->getPPALForReadAndWrite();

    for (unsigned int i=0; i<size; ++i) {    
      for (unsigned int b=0; b<conf.cur_bunch_size; ++b)
	units_ptr[i] = the_rand->rand();
      units_ptr += conf.max_bunch_size;
    }
  }
  
  //////////////////////////////////////////////////////////////////////////////
  
  ActivationFunction *getActivationFunctionByTypeString(const constString &str) {
    if (str == "inputs" || str ==  "linear")
      return new LinearActivationFunction();
    else if (str == "logistic") return new LogisticActivationFunction();
    else if (str == "tanh") return new TanhActivationFunction();
    else if (str == "softsign") return new SoftsignActivationFunction();
    else if (str == "softmax") return new SoftmaxActivationFunction();
    else ERROR_EXIT(256, "Incorrect activation function type\n");
    return 0;
  }

  ActivationFunction *getActivationFunctionByTypeString(const char *str) {
    constString cstr(str);
    return getActivationFunctionByTypeString(cstr);
  }
}
#undef sigmoid
