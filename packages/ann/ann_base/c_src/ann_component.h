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
#include "ann_configuration.h"
#include "aux_hash_table.h" // required during cloning process
#include "hash_table.h"     // required during cloning process
using april_utils::hash;    // required during cloning process

namespace ANN {

  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class ANNComponent : public Referenced {
    /// The name identifies this component to do fast search. It is a unique
    /// name, repetitions are forbidden.
    string name;
    string weights_name;
    unsigned int input_size;
    unsigned int output_size;
  public:
    ANNComponent(const char *name, const char *weights_name,
		 unsigned int input_size = 0, unsigned int output_size = 0) :
      name(name), weights_name(weights_name) { }
    virtual ~ANNComponent() { }

    virtual const Token *getInput() const       = 0;
    virtual const Token *getOutput() const      = 0;
    virtual const Token *getErrorInput() const  = 0;
    virtual const Token *getErrorOutput() const = 0;
    
    /// Abstract method that executes the set of operations required for each
    /// block of connections when performing the forward step of the
    /// Backpropagation algorithm, and returns its output Token
    virtual Token *doForward(Token* input, bool during_training) = 0;

    /// Abstract method that back-propagates error derivatives and computes
    /// other useful stuff. Receives input error gradients, and returns its
    /// output error gradients Token.
    virtual Token *doBackprop(Token *input_error) = 0;
    
    /// Abstract method that update weights given gradients and input/output
    /// data
    virtual void doUpdate() = 0;
    /// Abstract method to reset to zero gradients and outputs (inputs are not
    /// reseted)
    virtual void reset() = 0;
    
    virtual ANNComponent *clone() = 0;

    /// Virtual method for setting the value of a training parameter.
    virtual void setOption(const char *name, double value) { }

    /// Virtual method for determining if a training parameter
    /// is being used or can be used within the network.
    virtual bool hasOption(const char *name) { return false; }
    
    /// Virtual method for getting the value of a training parameter.
    virtual double getOption(const char *name) { return 0.0; }
    
    /// Abstract method to finish building of component hierarchy and set
    /// weights objects pointers
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict) = 0;
    
    /// Abstract method to retrieve Connections objects from ANNComponents
    virtual void copyWeights(hash<string,Connections*> &weights_dict) = 0;

    /// Abstract method to retrieve ANNComponents objects
    virtual void copyComponents(hash<string,ANNComponent*> &weights_dict) = 0;
    
    /// Virtual method which returns the component with the given name if
    /// exists, otherwise it returns 0. By default, base components only
    /// contains itself. Component composition will need to look to itself and
    /// all contained components.
    virtual ANNComponent *getComponent(string &name) {
      if (name == name) return this;
      return 0;
    }
  };
}

#endif // ACTION_H
