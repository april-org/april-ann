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

#ifndef FUNCTION_INTERFACE_H
#define FUNCTION_INTERFACE_H

#include "logbase.h"
#include "error_print.h"
#include "referenced.h"

namespace Functions {

  namespace NS_function_io {
    /// Enum which define input/output types for functions in April.
    enum type {
      FLOAT=0, LOGFLOAT, DOUBLE, UNKNOWN,
    };
  }
  
  /// A pure abstract templatized class that serves as interface.
  /**
     A DataProducer is an abstraction of an object which produces a sequence a
     vectors of a given type, followed by a 0 indicating sequence ending. The
     object could be reinitialized (if it is possible) using the reset
     method. Finally, it could be destroyed using destroy method.
     
     Destroy method is needed to make feasible the destruction of a DataProducer
     without calling its destructor, forcing the object to send a 0 in the next
     get() call.
   */
  template<typename T>
  class DataProducer : public Referenced {
  public:
    /// Returns a pointer and its property. A 0 inidicates data sequence ending.
    virtual T *get()     = 0;
    /// Reinitialize the sequence iterator if it is possible.
    virtual void reset() = 0;
    /// It destroys the object content and forces to return a 0 on next get().
    virtual void destroy() { }
  };

  /// A pure abstract templatized class that serves as interface.
  /**
     A DataConsumer is an abstraction of an object which consumes a sequence of
     vectors of a given type, followed by a 0 indicating sequence ending. The
     object could be reinitialized (if it is possible) using the reset
     method. Finally, it could be destroyed using destroy method.
     
     Destroy method is needed to make feasible the destruction of a DataConsumer
     without calling its destructor, forcing the object to send a 0 in the next
     get() call.
   */  
  template<typename T>
  class DataConsumer : public Referenced {
  public:
    /// Receives a pointer and its property. It receive a 0 at sequence ending.
    virtual void put(T *) = 0;
    /// Reinitialize the sequence iterator if it is possible.
    virtual void reset()  = 0;
    /// It destroys the object content.
    virtual void destroy() { }
  };
  
  /// A pure abstract class that serves as high level interface.
  /**
     A FunctionBase is an abstraction of an object which represents a
     mathematical function. It has an input/output type and an input/output
     size. This abstract class is used to store high level methods, which are
     independent from the input/output types. Every function is feeded with an
     input vector and produces an output vector.
     
     This class needs to be extended as a template (FunctionInterface class) in
     order to fix the input/output data types.
   */
  class FunctionBase : public Referenced {
  public:
    FunctionBase() : Referenced() { }
    virtual ~FunctionBase() {
    }
    /// It returns the data type of the input (or domain).
    virtual NS_function_io::type getInputType() const = 0;
    /// It returns the data type of the output (or range).
    virtual NS_function_io::type getOutputType() const = 0;
    /// It returns the input (or domain) size of the function.
    virtual unsigned int getInputSize()  const = 0;
    /// It returns the output (or range) size of the function.
    virtual unsigned int getOutputSize() const = 0;
  };
  
  /// A virtual templatized class which implement some basic functionality and describe methods for functions.
  /**
     A FunctionInterface is an abstraction of an object which represents a
     mathematical function, as a especialization of the FunctionBase class. It
     adds to the interface one abstract method which calculates output vector
     given input vector, and a basic implementation of a method that consumes
     input vectors (from a DataProducer) and produces output vectros (to a
     DataConsumer).
     
     This class needs to be instantiated, fixing input/output types, and needs
     to be extended in order to implement abstract methods.
   */  
  template<typename I, typename O>
  class FunctionInterface : public FunctionBase {
  public:
    virtual ~FunctionInterface() {
    }
    // Parent abstract methods 
    // virtual unsigned int getInputSize()  const = 0;
    // virtual unsigned int getOutputSize() const = 0;
    
    /// A new abstract method that computes output vector given input vector.
    virtual bool calculate(const I *input_vector, unsigned int input_size,
			   O *output_vector, unsigned int output_size) = 0;
    

    /// Method for flow computation of outputs in a pipeline system.
    /**
       This method process a flow of data vectors, computed by a DataProducer,
       and produces an output vector for each of them. Output vectors are given
       to a DataConsumer. The end of the flow is indicated by DataProducer using
       a 0 pointer. Is mandatory to produces a 0 pointer to feed the
       DataConsumer when the flow ends. Input vector pointer is property of this
       method, needs to be freed. Output vector pointer is property of the
       DataConsumer object.
     */
    virtual void calculateInPipeline(DataProducer<I> *producer,
				     unsigned int input_size,
				     DataConsumer<O> *consumer,
				     unsigned int output_size) {
      unsigned int _input_size, _output_size;
      _input_size  = getInputSize();
      _output_size = getOutputSize();
      if (_input_size != input_size) {
	ERROR_PRINT("Incorrect input size!!!\n");
      exit(128);
      }
      if (_output_size != input_size) {
	ERROR_PRINT("Incorrect output size!!!\n");
	exit(128);
      }
      float *input;
      // we read from producer input until a 0
      while( (input = producer->get()) != 0) {
	float *output = new float[output_size];
	// computes the output vector given the input
	calculate(input, input_size, output, output_size);
	// we lose the output pointer property after the put
	consumer->put(output);
	// we have the input pointer property, it is deleted
	delete[] input;
      }
      // is mandatory to send this 0 to the consumer when the flow of data ends
      consumer->put(0);
    }
  };
  
  /// Instantiation of a FunctionInterface with input=float and output=float
  typedef FunctionInterface<float,float>      FloatFloatFunctionInterface;

  /// Instantiation of a DataConsumer of floats
  typedef DataConsumer<float>                 FloatDataConsumer;
  /// Instantiation of a DataProducer of floats
  typedef DataProducer<float>                 FloatDataProducer;
  /// Instantiation of a DataConsumer of log_floats
  typedef DataConsumer<log_float>             LogFloatDataConsumer;
  /// Instantiation of a DataProducer of log_floats
  typedef DataProducer<log_float>             LogFloatDataProducer;
  /// Instantiation of a DataProducer of double
  typedef DataProducer<double>                DoubleDataProducer;
}

#endif //FUNCTION_INTERFACE_H
