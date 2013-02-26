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
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "error_print.h"
#include "MersenneTwister.h"
#include "constString.h"
#include "referenced.h"
#include "gpu_mirrored_memory_block.h"
#include "wrapper.h"
#include "ann_configuration.h"

namespace ANN {

  /// A pure abstract class that define the basic interface that
  /// activation functions must fulfill.
  class ActivationFunction : public Referenced
  {
  public:
    virtual ~ActivationFunction() { }

    /// Abstract method that activates a "bunch" of units (or neurons).
    /**
       Abstract method that takes the input neuron potentials and computes the
       activation:
     
       \f[
       a_i = f(y_i | y_1, y_2, \ldots y_n)
       \f]
       
       being \f$a_i\f$ the activation of neuron \f$i\f$, depending on its
       potential \f$y_i\f$ and the other neurons potentials.
       
       @param units an input matrix of size (size X conf.cur_bunch_size). It
       is stored at an FloatGPUMirroredMemoryBlock object, Each component is a
       neuron activation.
       
       @param units_size the number of neurons.

       @param conf a reference to an ANNConfiguration struct which contains
       global parameters as: current bunch size, maximum bunch size, use_cuda
       global flag, \ldots
       
       @param use_cuda a boolean parameter that indicates the use of CPU
       computation or GPU computation (using CUDA). Overrides the configuration
       stored at conf parameter.
     */
    virtual void applyActivation(FloatGPUMirroredMemoryBlock *units, 
				 unsigned int units_size,
				 const ANNConfiguration &conf,
				 bool use_cuda) = 0;

    /// Abstract method that computes derivatives.
    /** Abstract method that computes the derivative of a "bunch" of units (or
      neurons) and stores it on input_errors vector. Basically, it computes the
      following equation:
      
      \f[
      \delta_i = \delta_i \cdot \frac{\partial a_i}{\partial y_i}
      \f]
      
      @param units an input matrix of size (size X conf.cur_bunch_size). It
      is stored at an FloatGPUMirroredMemoryBlock object, Each component is a
      neuron activation.

      @param input_errors an input matrix of size (size X
      conf.cur_bunch_size). It is stored at an FloatGPUMirroredMemoryBlock
      object. Each component is a sum of error derivatives that comes from the
      output layer neurons to the current neuron.
      
      @param size an unsigned int parameter indicating the number of
      neurons.
      
      @param conf an ANNConfiguration reference which stores some global
      parameters as: current bunch size, maximum bunch size, use_cuda global
      flag, ...
      
      @param use_cuda a boolean flag indicating the use of CPU computation or
      GPU computation (using CUDA). Overrides the configuration stored at conf
      parameter.
      
    */
    virtual void multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
				     FloatGPUMirroredMemoryBlock *input_errors,
				     unsigned int size,
				     const ANNConfiguration &conf,
				     bool use_cuda,
				     bool is_output) = 0;
    
    /// Returns the minium value
    virtual float getDropoutValue() = 0;
    
    
    /// Returns a deep copy of the object
    virtual ActivationFunction *clone() = 0;
  };

  //////////////////////////////////////////////////////////
  
  //! A logistic (or sigmoid) activation function.
  /*! Implements ActivationFunction interface and, given a neuron potential y,
      it follows the equation:

     \f[
     a_i = f(y_i) = \frac{1}{1 + e^{-y_i}}
     \f]
  */
  class LogisticActivationFunction : public ActivationFunction
  {
  public:
    virtual ~LogisticActivationFunction() { }
    void applyActivation(FloatGPUMirroredMemoryBlock *units, unsigned int units_size,
			 const ANNConfiguration &conf, bool use_cuda);
    
    void multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
			     FloatGPUMirroredMemoryBlock *input_errors,
			     unsigned int size,
			     const ANNConfiguration &conf,
			     bool use_cuda,
			     bool is_output);
    float getDropoutValue() { return 0.0f; }
    ActivationFunction *clone();
  };

  //////////////////////////////////////////////////////////

  //! A tanh activation function.
  /*! Implements ActivationFunction interface and, given a neuron potential y,
      follows the equation:

     \f[
     a_i = f(y_o) = \frac{2}{1 + e^{-y_i}} - 1
     \f]
  */
  class TanhActivationFunction : public ActivationFunction
  {
  public:
    virtual ~TanhActivationFunction() { }
    void applyActivation(FloatGPUMirroredMemoryBlock *units, unsigned int units_size,
			 const ANNConfiguration &conf, bool use_cuda);
    void multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
			     FloatGPUMirroredMemoryBlock *input_errors,
			     unsigned int size,
			     const ANNConfiguration &conf,
			     bool use_cuda,
			     bool is_output);
    float getDropoutValue() { return 0.0f; }
    ActivationFunction *clone();
  };

  //////////////////////////////////////////////////////////

  //! A softmax activation function.
  /*! Implements ActivationFunction interface and, given a bunch_size number of
    neuron layers, follows the equation:

     \f[
     a_i = \frac{e^{y_i}}{\sum_{j} e^{y_j}}
     \f]
     
     If use_cuda flag is true, the object reserves memory for the map-reduce
     process executed at the GPU. Therefore, with use_cuda=true the object is
     unable to be shared between different ActivationUnits.
  */
  class SoftmaxActivationFunction : public ActivationFunction
  {
    /// If use_cuda=true, this is the size of the ActivationUnits.
    unsigned int size;
    
    FloatGPUMirroredMemoryBlock
    *minimums,   ///< Only with use_cuda=true, for map-reduce process.
      *maximums, ///< Only with use_cuda=true, for map-reduce process.
      *sums;     ///< Only with use_cuda=true, for map-reduce process.
  public:
    SoftmaxActivationFunction();
    virtual ~SoftmaxActivationFunction();
    void applyActivation(FloatGPUMirroredMemoryBlock *units, 
                         unsigned int units_size,
			 const ANNConfiguration &conf,
                         bool use_cuda);
    void multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
			     FloatGPUMirroredMemoryBlock *input_errors,
			     unsigned int size,
			     const ANNConfiguration &conf,
			     bool use_cuda,
			     bool is_output);
    float getDropoutValue() { return 0.0f; }
    ActivationFunction *clone();
  };


  //! A linear activation function.
  /*! Implements ActivationFunction interface but it doesn't compute nothing. It
      would be necessary in some future scenarios, nevertheless it is not
      computing anything at this moment.
  */
  class LinearActivationFunction : public ActivationFunction
  {
  public:
    LinearActivationFunction();
    virtual ~LinearActivationFunction();
    void applyActivation(FloatGPUMirroredMemoryBlock *units, 
                         unsigned int units_size,
			 const ANNConfiguration &conf,
                         bool use_cuda);
    void multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
			     FloatGPUMirroredMemoryBlock *input_errors,
			     unsigned int size,
			     const ANNConfiguration &conf,
			     bool use_cuda,
			     bool is_output);
    float getDropoutValue() { return 0.0f; }
    ActivationFunction *clone();
  };

  //////////////////////////////////////////////////////////

  /// A virtual class that define the basic interface for stochastic activation
  /// functions. That is, activation functions that are based in a stochastic
  /// process.
  class StochasticActivationFunction : public ActivationFunction {
  protected:
    /// A pseudo-random generator object for the stochastic process.
    MTRand *the_rand;
    
    /// A method that samples 1.0 or 0.0 with probability v.
    /*!
      @param v a float which indicating the probability of being 1.0
    */
    float sampleOne(float v) {
      return (the_rand->rand() < v) ? 1.0f: 0.0f;
    }
  public:
    /// A constructor.
    /*!
      @param the_rand a MTRand object for pseudo-random numbers generation
    */
    StochasticActivationFunction(MTRand *the_rand) : the_rand(the_rand) {
      IncRef(the_rand);
    }
    virtual ~StochasticActivationFunction() {
      DecRef(the_rand);
    }
    
    /// An abstract method for apply the stochastic process over all neurons.
    /*!

      @param units a memory matrix of size (size X bunch_size), represented in a
      FloatGPUMirroredMemoryBlock object. Its values are the neuron
      probabilities of being 1.0.

      @param size an unsigned int parameter indicating the number of neurons on
      the corresponding layer.

      @param bunch_size an unsigned int parameter indicating the number of
      layers. Each layer is composed by size neurons, and each layer comes from
      the computation of the forward of a different input pattern.
      
      @param use_cuda a boolean parameter to switch between CPU computation or
      GPU computation (using CUDA).

     */
    virtual void randomize(FloatGPUMirroredMemoryBlock *units,
			   unsigned int size,
			   const ANNConfiguration &conf,
			   bool use_cuda) = 0;
    
    /// An abstract method for compute derivatives of stochastic activation functions
    /*!
      
      @param units a memory matrix of size (size X bunch_size), represented in a
      FloatGPUMirroredMemoryBlock object. Its values are the neuron
      probabilities of being 1.0.

      @param inputs_errors a memory matrix of size (size X bunch_size),
      represented in a FloatGPUMirroredMemoryBlock object. Its values are the
      neuron probabilities derivative of the error that comes from the output.
      
      @param size an unsigned int parameter indicating the number of neurons on
      the corresponding layer.

      @param conf a reference to an ANNConfiguration object that contains global
      parameters as: current bunch size, maximum bunch size, use_cuda global
      flag, ...
      
      @param use_cuda a boolean parameter to switch between CPU computation or
      GPU computation (using CUDA). Overrides the configuration stored at conf
      parameter.
      
    */
    void multiplyDerivatives(FloatGPUMirroredMemoryBlock *units,
			     FloatGPUMirroredMemoryBlock *input_errors,
			     unsigned int size,
			     const ANNConfiguration &conf,
			     bool use_cuda,
			     bool is_output) {
      ERROR_PRINT("NOT SUPORTTED METHOD");
    }
    float getDropoutValue() { return 0.0f; }
  };
  
  /// A binary sampling activation function. Implements
  /// ActivationFunction interface. This is an special activation
  /// function that works on stochastic graphical models as Restricted
  /// Boltzmann Machines.
  class BinarySamplingActivationFunction :
    public StochasticActivationFunction {
  public:
    BinarySamplingActivationFunction(MTRand *the_rand);
    virtual ~BinarySamplingActivationFunction();
    void applyActivation(FloatGPUMirroredMemoryBlock *units, unsigned int units_size,
			 const ANNConfiguration &conf, bool use_cuda);

    ActivationFunction *clone();
    void randomize(FloatGPUMirroredMemoryBlock *units,
		   unsigned int size,
		   const ANNConfiguration &conf,
		   bool use_cuda);
  };
  
  //////////////////////////////////////////////////////////

  /// Auxiliar function to build ActivationFunction objects.
  /*!
    Auxiliar function to build an ActivationFunction instance from
    its name string (as a char*). It only works on simple activation functions
    that doesn't have parametrized constructor.
  */
  ActivationFunction *getActivationFunctionByTypeString(const char *str);

  /// Auxiliar function to build ActivationFunction objects.
  /*!
    Auxiliar function to build an ActivationFunction instance from
    its name string (as a constString&). It only works on
    simple activation functions that doesn't have parametrized
    constructor.
  */
  ActivationFunction *getActivationFunctionByTypeString(const constString &str);
}

#endif // ACTIVATION_FUNCTION_H
