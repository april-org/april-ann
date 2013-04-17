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
#ifndef ANNCOMPONENT_H
#define ANNCOMPONENT_H

#include "string.h"
#include "constString.h"
#include "actunit.h"
#include "connection.h"
#include "referenced.h"
#include "error_print.h"
#include "aux_hash_table.h" // required for build
#include "hash_table.h"     // required for build
using april_utils::hash;    // required for build

namespace ANN {

  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class ANNComponent : public Referenced {
  protected:
    /// The name identifies this component to do fast search. It is a unique
    /// name, repetitions are forbidden.
    string name;
    string weights_name;
    unsigned int input_size;
    unsigned int output_size;
    bool use_cuda;
  public:
    ANNComponent(const char *name, const char *weights_name = 0,
		 unsigned int input_size = 0, unsigned int output_size = 0) :
      name(name), use_cuda(false) {
      if (weights_name) this->weights_name = string(weights_name);
    }
    virtual ~ANNComponent() { }
    
    unsigned int getInputSize() const { return input_size; }
    unsigned int getOutputSize() const { return output_size; }
    
    virtual const Token *getInput() const       = 0;
    virtual const Token *getOutput() const      = 0;
    virtual const Token *getErrorInput() const  = 0;
    virtual const Token *getErrorOutput() const = 0;
    
    /// Virtual method that executes the set of operations required for each
    /// block of connections when performing the forward step of the
    /// Backpropagation algorithm, and returns its output Token
    virtual Token *doForward(Token* input, bool during_training) {
      return input;
    }

    /// Virtual method that back-propagates error derivatives and computes
    /// other useful stuff. Receives input error gradients, and returns its
    /// output error gradients Token.
    virtual Token *doBackprop(Token *input_error) {
      return input_error;
    }
    
    /// Virtual method that update weights given gradients and input/output
    /// data
    virtual void doUpdate() { }
    /// Virtual method to reset to zero gradients and outputs (inputs are not
    /// reseted)
    virtual void reset() { }
    
    virtual ANNComponent *clone() = 0;
    
    virtual void setUseCuda(bool v) { use_cuda = true; }
    
    /// Virtual method for setting the value of a training parameter.
    virtual void setOption(const char *name, double value) { }

    /// Virtual method for determining if a training parameter
    /// is being used or can be used within the network.
    virtual bool hasOption(const char *name) { return false; }
    
    /// Virtual method for getting the value of a training parameter.
    virtual double getOption(const char *name) { return 0.0; }
    
    /// Abstract method to finish building of component hierarchy and set
    /// weights objects pointers
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict) {
      ////////////////////////////////////////////////////////////////////
      ANNComponent *&component = components_dict[name];
      if (component != 0) ERROR_EXIT(102, "Non unique component name found: %s\n",
				     name.c_str());
      component = this;
      ////////////////////////////////////////////////////////////////////
      if (input_size == 0)  input_size  = _input_size;
      if (output_size == 0) output_size = _output_size;
      if (input_size != _input_size)
	ERROR_EXIT2(129, "Incorrect input size, expected %d, found %d\n",
		    input_size, _input_size);
      if (output_size != _output_size)
	ERROR_EXIT2(129, "Incorrect output size, expected %d, found %d\n",
		    output_size, _output_size);
    }
    
    /// Abstract method to retrieve Connections objects from ANNComponents
    virtual void copyWeights(hash<string,Connections*> &weights_dict) = 0;

    /// Abstract method to retrieve ANNComponents objects
    virtual void copyComponents(hash<string,ANNComponent*> &weights_dict) {
      components_dict[name] = this;
    }
    
    /// Virtual method which returns the component with the given name if
    /// exists, otherwise it returns 0. By default, base components only
    /// contains itself. Component composition will need to look to itself and
    /// all contained components.
    virtual ANNComponent *getComponent(string &name) {
      if (name == name) return this;
      return 0;
    }
    
    /// Final method (FIXME: review for C++11 standard), computes fan in/out for
    /// a given weights_name, adding input/output size when apply
    virtual void computeFanInAndFanOut(const string &weights_name,
			       unsigned int &fan_in,
			       unsigned int &fan_out) {
      if (this->weights_name && weights_name == this->weights_name) {
	fan_in  += input_size;
	fan_out += output_size;
      }
    }
  };
}

#endif // ANNCOMPONENT_H
