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
#include "token_base.h"

namespace Functions {
  
  /// A pure abstract class that serves as interface.
  /**
     A DataProducer is an abstraction of an object which produces a sequence a
     vectors of a given type, followed by a 0 indicating sequence ending. The
     object could be reinitialized (if it is possible) using the reset
     method. Finally, it could be destroyed using destroy method.
     
     Destroy method is needed to make feasible the destruction of a DataProducer
     without calling its destructor, forcing the object to send a 0 in the next
     get() call.
   */
  class DataProducer : public Referenced {
  public:
    /// Returns a pointer and its property. A 0 inidicates data sequence ending.
    virtual Token *get()     = 0;
    /// Reinitialize the sequence iterator if it is possible.
    virtual void reset() = 0;
    /// It destroys the object content and forces to return a 0 on next get().
    virtual void destroy() { }
  };

  /// A pure abstract class that serves as interface.
  /**
     A DataConsumer is an abstraction of an object which consumes a sequence of
     vectors of a given type, followed by a 0 indicating sequence ending. The
     object could be reinitialized (if it is possible) using the reset
     method. Finally, it could be destroyed using destroy method.
     
     Destroy method is needed to make feasible the destruction of a DataConsumer
     without calling its destructor, forcing the object to send a 0 in the next
     get() call.
   */  
  class DataConsumer : public Referenced {
  public:
    /// Receives a pointer and its property. It receive a 0 at sequence ending.
    virtual void put(Token *) = 0;
    /// Reinitialize the sequence iterator if it is possible.
    virtual void reset()  = 0;
    /// It destroys the object content.
    virtual void destroy() { }
  };
  
  /// A virtual class that serves as high level interface.
  /**
     A FunctionInterface is an abstraction of an object which represents a
     mathematical function. It adds to the interface abstract methods which
     calculates output vector given input vector, and a basic implementation of
     a method that consumes input vectors (from a DataProducer) and produces
     output vectros (to a DataConsumer).  Every function is feeded with an input
     Token and produces an output Token.
   */
  class FunctionInterface : public Referenced {
  public:
    FunctionInterface() : Referenced() { }
    virtual ~FunctionInterface() {
    }
    /// It returns the input (or domain) size of the function.
    virtual unsigned int getInputSize()  const = 0;
    /// It returns the output (or range) size of the function.
    virtual unsigned int getOutputSize() const = 0;
    /// A new abstract method that computes output vector given input vector.
    virtual Token *calculate(const Token *input_vector) = 0;
    

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
    virtual void calculateInPipeline(DataProducer *producer,
				     DataConsumer *consumer) {
      Token *input;
      // we read from producer input until a 0
      while( (input = producer->get()) != 0) {
	// we get the input increasing its reference counter
	IncRef(input);
	Token *output = calculate(input);
	// we get the output increasing its reference counter
	IncRef(output);
	consumer->put(output);
	// we lose input and output decreasing reference counters
	DecRef(output);
	DecRef(input);
      }
      // is mandatory to send this 0 to the consumer when the flow of data ends
      consumer->put(0);
    }
  };
}

#endif //FUNCTION_INTERFACE_H
