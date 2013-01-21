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
#ifndef ERROR_FUNC_H
#define ERROR_FUNC_H

#include "actunit.h"
#include "referenced.h"

namespace ANN {

  class ErrorFunction : public Referenced {
  public:
    virtual ~ErrorFunction() { }
  
    virtual void computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
					     FloatGPUMirroredMemoryBlock *target_output,
					     FloatGPUMirroredMemoryBlock *output_error,
					     FloatGPUMirroredMemoryBlock *pattern_errors,
					     unsigned int output_size,
					     const ANNConfiguration &conf) = 0;
    virtual float computeBatchErrorFunction(float error_sums,
					    unsigned int num_patterns)  = 0;
    virtual ErrorFunction *clone() = 0; // deep copy
    virtual bool logBaseComputation() { return false; }
  };

  class MSE : public ErrorFunction {
    const float zero_epsilon_distance;
  public:
    MSE();
    virtual ~MSE() { }
  
    // calcula el MSE
    void computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
				     FloatGPUMirroredMemoryBlock *target_output,
				     FloatGPUMirroredMemoryBlock *output_error,
				     FloatGPUMirroredMemoryBlock *pattern_errors,
				     unsigned int output_size,
				     const ANNConfiguration &conf);
    float computeBatchErrorFunction(float error_sums,
				    unsigned int num_patterns);
    ErrorFunction *clone() {
      return new MSE();
    }
  };

  /////////////////////////////////////////////////////

  class Tanh : public ErrorFunction {
  public:
    Tanh();
    virtual ~Tanh() { }
  
    // calcula el MSE
    void computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
				     FloatGPUMirroredMemoryBlock *target_output,
				     FloatGPUMirroredMemoryBlock *output_error,
				     FloatGPUMirroredMemoryBlock *pattern_errors,
				     unsigned int output_size,
				     const ANNConfiguration &conf);
    float computeBatchErrorFunction(float error_sums,
				    unsigned int num_patterns);
    ErrorFunction *clone() {
      return new Tanh();
    }
  };
  
  /*
//////////////////////////////////////////////////// 

class GA : public ErrorFunction {
public:
GA();
virtual ~GA() {}
float computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
FloatGPUMirroredMemoryBlock *target_output,
FloatGPUMirroredMemoryBlock *output_error,
FloatGPUMirroredMemoryBlock *pattern_errors,
unsigned int output_size,
const ANNConfiguration &conf);
float computeBatchErrorFunction(float error_sums,
unsigned int num_patterns);
ErrorFunction *clone() {
return new GA();
}
};

  */
  ///////////////////////////////////////
  
  class CrossEntropy : public ErrorFunction {
    const float EPSILON, INF;
  public:
    CrossEntropy();
    virtual ~CrossEntropy() {}

    // calcula la entropia cruzada
    void computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
				     FloatGPUMirroredMemoryBlock *target_output,
				     FloatGPUMirroredMemoryBlock *output_error,
				     FloatGPUMirroredMemoryBlock *pattern_errors,
				     unsigned int output_size,
				     const ANNConfiguration &conf);
    float computeBatchErrorFunction(float error_sums,
				    unsigned int num_patterns);
    ErrorFunction *clone() {
      return new CrossEntropy();
    }
  };

  ///////////////////////////////////////
  

  // G(a,b) = 2 * \sum_i a_i b_i
  // H(a,b) = \sum_i ( b_i + a_i )
  //
  // F'(a,b)/a_i = \alpha ( 2 b_i H(a,b) - G(a,b) ) / H^2(a,b)
  class LocalFMeasure : public ErrorFunction {
    float alpha;
    unsigned int N;
  public:
    LocalFMeasure(float alpha = 2.0f);
    virtual ~LocalFMeasure() {}
    
    void computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
				     FloatGPUMirroredMemoryBlock *target_output,
				     FloatGPUMirroredMemoryBlock *output_error,
				     FloatGPUMirroredMemoryBlock *pattern_errors,
				     unsigned int output_size,
				     const ANNConfiguration &conf);
    float computeBatchErrorFunction(float error_sums,
				    unsigned int num_patterns);
    ErrorFunction *clone() {
      return new LocalFMeasure(alpha);
    }
  };
  
  ///////////////////////////////////////
  
  class FullCrossEntropy : public ErrorFunction {
    const float EPSILON, INF;
  public:
    FullCrossEntropy();
    virtual ~FullCrossEntropy() {}

    // calcula la entropia cruzada
    void computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
				     FloatGPUMirroredMemoryBlock *target_output,
				     FloatGPUMirroredMemoryBlock *output_error,
				     FloatGPUMirroredMemoryBlock *pattern_errors,
				     unsigned int output_size,
				     const ANNConfiguration &conf);
    float computeBatchErrorFunction(float error_sums,
				    unsigned int num_patterns);
    ErrorFunction *clone() {
      return new FullCrossEntropy();
    }
    bool logBaseComputation() { return true; }
  };

  ///////////////////////////////////////

  /*
    class MixtureCrossEntropy : public ErrorFunction {
    const float EPSILON, INF;
    public:
    MixtureCrossEntropy();
    virtual ~MixtureCrossEntropy() {}

    // calcula la entropia cruzada, target_output 
    float computePatternErrorFunction(FloatGPUMirroredMemoryBlock *output,
    FloatGPUMirroredMemoryBlock *target_output,
    FloatGPUMirroredMemoryBlock *output_error,
    FloatGPUMirroredMemoryBlock *pattern_errors,
    unsigned int output_size,
    const ANNConfiguration &conf);
    float computeBatchErrorFunction(float error_sums,
    unsigned int num_patterns);
    ErrorFunction *clone() {
    return new MixtureCrossEntropy();
    }
    };
  */
  
  /////////////////////////////////////////////////////

  class TimeSeriesErrorFunction : public Referenced {
  public:
    virtual ~TimeSeriesErrorFunction() { }
    virtual float computeErrorFromTimeSerie(//FloatGPUMirroredMemoryBlock *output,
					    //FloatGPUMirroredMemoryBlock *target_output,
					    float *output,
					    float *target_output,
					    unsigned int output_size) = 0;
  };
  
  // only for off-line use, not during training
  class NormalizedRootMSE : public TimeSeriesErrorFunction {
  public:
    ~NormalizedRootMSE();
    float computeErrorFromTimeSerie(//FloatGPUMirroredMemoryBlock *output,
				    //FloatGPUMirroredMemoryBlock *target_output,
				    float *output,
				    float *target_output,
				    unsigned int output_size);
  };

  class RootMSE : public TimeSeriesErrorFunction {
  public:
    ~RootMSE();
    float computeErrorFromTimeSerie(//FloatGPUMirroredMemoryBlock *output,
				    //FloatGPUMirroredMemoryBlock *target_output,
				    float *output,
				    float *target_output,
				    unsigned int output_size);
  };

  // only for off-line use, not during training
  class AbsoluteError : public TimeSeriesErrorFunction {
  public:
    ~AbsoluteError();
    float computeErrorFromTimeSerie(//FloatGPUMirroredMemoryBlock *output,
				    //FloatGPUMirroredMemoryBlock *target_output,
				    float *output,
				    float *target_output,
				    unsigned int output_size);
  };
}

#endif // ERROR_FUNC_H
