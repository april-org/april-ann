/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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

#include <cstring>
#include "mystring.h"
#include "connection.h"
#include "referenced.h"
#include "error_print.h"
#include "token_base.h"
#include "aux_hash_table.h" // required for build
#include "hash_table.h"     // required for build
using april_utils::hash;    // required for build
using april_utils::string;

#define MAX_NAME_STR 120

#define mSetOption(var_name,var) if(!strcmp(name,(var_name))){(var)=value;return;}
#define mHasOption(var_name) if(!strcmp(name,(var_name))) return true;
#define mGetOption(var_name, var) if(!strcmp(name,(var_name)))return (var);

namespace ANN {

  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class ANNComponent : public Referenced {
  private:
    bool is_built;
    void generateDefaultName() {
      char str_id[MAX_NAME_STR+1];
      snprintf(str_id, MAX_NAME_STR, "c%u", next_name_id);
      name = string(str_id);
      ++next_name_id;
    }
  protected:
    static unsigned int next_name_id;
    static unsigned int next_weights_id;
    /// The name identifies this component to do fast search. It is a unique
    /// name, repetitions are forbidden.
    string name;
    string weights_name;
    unsigned int input_size;
    unsigned int output_size;
    bool use_cuda;
  public:
    ANNComponent(const char *name = 0, const char *weights_name = 0,
		 unsigned int input_size = 0, unsigned int output_size = 0) :
      Referenced(),
      is_built(false),
      input_size(input_size), output_size(output_size),
      use_cuda(false) {
      if (name) this->name = string(name);
      else generateDefaultName();
      if (weights_name) this->weights_name = string(weights_name);
    }
    virtual ~ANNComponent() { }

    const string &getName() const { return name; }
    const string &getWeightsName() const { return weights_name; }
    bool hasWeightsName() const { return !weights_name.empty(); }
    
    bool getIsBuilt() const { return is_built; }
    
    void generateDefaultWeightsName() {
      char str_id[MAX_NAME_STR+1];
      snprintf(str_id, MAX_NAME_STR, "w%u", next_weights_id);
      weights_name = string(str_id);
      ++next_weights_id;
    }

    unsigned int getInputSize() const {
      return input_size;
    }
    unsigned int getOutputSize() const {
      return output_size;
    }
    
    virtual Token *getInput() { return 0; }
    virtual Token *getOutput() { return 0; }
    virtual Token *getErrorInput() { return 0; }
    virtual Token *getErrorOutput() { return 0; }
    
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
    
    virtual ANNComponent *clone() {
      return new ANNComponent(name.c_str(), weights_name.c_str(),
			      input_size, output_size);
    }
    
    /// Virtual method to set use_cuda option. All childs which rewrite this
    /// method must call parent method before do anything.
    virtual void setUseCuda(bool v) { use_cuda = true; }
    
    /// Virtual method for setting the value of a training parameter.
    virtual void setOption(const char *name, double value) { }

    /// Virtual method for determining if a training parameter
    /// is being used or can be used within the network.
    virtual bool hasOption(const char *name) { return false; }
    
    /// Virtual method for getting the value of a training parameter. All childs
    /// which rewrite this method must call parent method after their process,
    /// if the looked option is not found to show the error message.
    virtual double getOption(const char *name) {
      ERROR_EXIT1(140, "The option %s does not exist.\n", name);
      return 0.0f;
    }
    
    /// Abstract method to finish building of component hierarchy and set
    /// weights objects pointers. All childs which rewrite this method must call
    /// parent method before do anything.
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict) {
      // if (is_built) ERROR_EXIT(128, "Rebuild is forbidden!!!!\n");
      is_built = true;
      ////////////////////////////////////////////////////////////////////
      ANNComponent *&component = components_dict[name];
      if (component != 0) ERROR_EXIT1(102, "Non unique component name found: %s\n",
				      name.c_str());
      component = this;
      ////////////////////////////////////////////////////////////////////
      if (input_size   == 0)  input_size   = _input_size;
      if (output_size  == 0)  output_size  = _output_size;
      if (_input_size  == 0)  _input_size  = input_size;
      if (_output_size == 0)  _output_size = output_size;
      if (input_size != _input_size)
	ERROR_EXIT2(129, "Incorrect input size, expected %d, found %d\n",
		    input_size, _input_size);
      if (output_size != _output_size)
	ERROR_EXIT2(129, "Incorrect output size, expected %d, found %d\n",
		    output_size, _output_size);
    }
    
    /// Abstract method to retrieve Connections objects from ANNComponents
    virtual void copyWeights(hash<string,Connections*> &weights_dict) { }

    /// Abstract method to retrieve ANNComponents objects. All childs which
    /// rewrite this method must call parent method before do anything.
    virtual void copyComponents(hash<string,ANNComponent*> &components_dict) {
      components_dict[name] = this;
    }
    
    /// 
    virtual void resetConnections() { }
    
    ///
    virtual void debugInfo() {
      fprintf(stderr, "Component '%s' ('%s')  %d inputs   %d outputs\n",
	      name.c_str(),
	      !(weights_name.empty())?weights_name.c_str():"(null)",
	      input_size, output_size);
    }
    
    /// Virtual method which returns the component with the given name if
    /// exists, otherwise it returns 0. By default, base components only
    /// contains itself. Component composition will need to look to itself and
    /// all contained components. All childs which rewrite this method must
    /// call parent method before do anything.
    virtual ANNComponent *getComponent(string &name) {
      if (this->name == name) return this;
      return 0;
    }
  };
}

#undef MAX_NAME_STR

#endif // ANNCOMPONENT_H
