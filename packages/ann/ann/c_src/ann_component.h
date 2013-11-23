/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#include "unused_variable.h"
#include "function_interface.h"
#include "unused_variable.h"
#include "mystring.h"
#include "connection.h"
#include "error_print.h"
#include "token_base.h"
#include "aux_hash_table.h" // required for build
#include "hash_table.h"     // required for build
#include "matrixFloat.h"

using april_utils::hash;    // required for build
using april_utils::string;

#ifndef NDEBUG
#define ASSERT_MATRIX(m) do {						\
    april_assert( (m)->getNumDim() == 2 );				\
    april_assert( (m)->getMajorOrder() == CblasColMajor );		\
  } while(0)
#else
#define ASSERT_MATRIX(m)
#endif

#define MAX_NAME_STR 256

#define LEARNING_RATE_STRING    "learning_rate"
#define MOMENTUM_STRING         "momentum"
#define WEIGHT_DECAY_STRING     "weight_decay"
#define DROPOUT_FACTOR_STRING   "dropout_factor"
#define DROPOUT_SEED_STRING     "dropout_seed"
#define MAX_NORM_PENALTY_STRING "max_norm_penalty"

#define mSetOption(var_name,var) if(!strcmp(name,(var_name))){(var)=value;return;}
#define mHasOption(var_name) if(!strcmp(name,(var_name))) return true;
#define mGetOption(var_name, var) if(!strcmp(name,(var_name)))return (var);

namespace ANN {
  
  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class ANNComponent : public Functions::FunctionInterface {
  private:
    bool is_built;
    void generateDefaultName(const char *prefix=0) {
      char default_prefix[2] = "c";
      char str_id[MAX_NAME_STR+1];
      if (prefix == 0) prefix = default_prefix;
      snprintf(str_id, MAX_NAME_STR, "%s%u", prefix, next_name_id);
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

    /// Method which computes the gradient of the weights on the given
    /// MatrixFloat object
    virtual void computeGradients(MatrixFloat*& weight_grads) {
      UNUSED_VARIABLE(weight_grads);
    }
    
  public:
    ANNComponent(const char *name = 0, const char *weights_name = 0,
		 unsigned int input_size = 0, unsigned int output_size = 0) :
      Functions::FunctionInterface(),
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
    
    static void resetIdCounters() { next_name_id=0; next_weights_id=0; }
    
    bool getIsBuilt() const { return is_built; }
    
    void generateDefaultWeightsName(string &dest,
				    const char *prefix=0) {
      char str_id[MAX_NAME_STR+1];
      char default_prefix[2] = "w";
      if (prefix == 0) prefix = default_prefix;
      snprintf(str_id, MAX_NAME_STR, "%s%u", prefix, next_weights_id);
      dest = string(str_id);
      ++next_weights_id;
    }
    
    // FunctionInterface methods
    virtual unsigned int getInputSize() const {
      return input_size;
    }
    virtual unsigned int getOutputSize() const {
      return output_size;
    }
    virtual Token *calculate(Token *input) {
      return this->doForward(input, false);
    }
    
    virtual Token *getInput() { return 0; }
    virtual Token *getOutput() { return 0; }
    virtual Token *getErrorInput() { return 0; }
    virtual Token *getErrorOutput() { return 0; }

    /// Virtual method that executes the set of operations required for each
    /// block of connections when performing the forward step of the
    /// Backpropagation algorithm, and returns its output Token
    virtual Token *doForward(Token* input, bool during_training) {
      UNUSED_VARIABLE(during_training);
      return input;
    }

    /// Virtual method that back-propagates error derivatives and computes
    /// other useful stuff. Receives input error gradients, and returns its
    /// output error gradients Token.
    virtual Token *doBackprop(Token *input_error) {
      return input_error;
    }
    
    /// Virtual method to reset to zero gradients and outputs (inputs are not
    /// reseted). It receives a counter of the number of times it is called by
    /// iterative optimizers (as conjugate gradient).
    virtual void reset(unsigned int it=0) {
      UNUSED_VARIABLE(it);
    }

    /// Method which receives a hash table with matrices where compute the
    /// gradients.
    virtual void computeAllGradients(hash<string,MatrixFloat*> &weight_grads_dict){
      if (!weights_name.empty())
	computeGradients(weight_grads_dict[weights_name]);
    }
    
    virtual ANNComponent *clone() {
      return new ANNComponent(name.c_str(), weights_name.c_str(),
			      input_size, output_size);
    }
    
    /// Virtual method to set use_cuda option. All childs which rewrite this
    /// method must call parent method before do anything.
    virtual void setUseCuda(bool v) {
#ifdef USE_CUDA
      use_cuda = v;
#else
      UNUSED_VARIABLE(v);
      ERROR_PRINT("WARNING!!! Trying to set use_cuda=true with NON "
		  "cuda compilation\n");
      use_cuda = false; // always false in this case
#endif
    }
    
    bool getUseCuda() const { return use_cuda; }
    
    /// Virtual method for setting the value of a training parameter.
    virtual void setOption(const char *name, double value) {
      UNUSED_VARIABLE(value);
      if (strcmp(name,LEARNING_RATE_STRING)==0 ||
	  strcmp(name,MOMENTUM_STRING)==0      ||
	  strcmp(name,WEIGHT_DECAY_STRING)==0  ||
	  strcmp(name,MAX_NORM_PENALTY_STRING)==0) {
	ERROR_EXIT(128, "DEPRECATED: learning_rate, momentum, weight_decay and "
		   "max_norm_penalty learning parameters are available "
		   "through ann.optimizer:set_option(...) objects, or via "
		   "trainable.supervised_trainer:set_option(...) method\n");
      }
      ERROR_EXIT1(140, "The option to be set does not exist: %s.\n", name);
    }

    /// Virtual method for determining if a training parameter
    /// is being used or can be used within the network.
    virtual bool hasOption(const char *name) { UNUSED_VARIABLE(name); return false; }
    
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
      UNUSED_VARIABLE(weights_dict);
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
      if (_output_size != 0)  output_size  = _output_size;
      if (_input_size  != 0)  input_size   = _input_size;
      if (input_size != _input_size)
	ERROR_EXIT2(129, "Incorrect input size, expected %d, found %d\n",
		    input_size, _input_size);
      if (output_size != _output_size)
	ERROR_EXIT2(129, "Incorrect output size, expected %d, found %d\n",
		    output_size, _output_size);
    }
    
    /// Abstract method to retrieve Connections objects from ANNComponents
    virtual void copyWeights(hash<string,Connections*> &weights_dict) {
      UNUSED_VARIABLE(weights_dict);
    }

    /// Abstract method to retrieve ANNComponents objects. All childs which
    /// rewrite this method must call parent method before do anything.
    virtual void copyComponents(hash<string,ANNComponent*> &components_dict) {
      components_dict[name] = this;
    }
    
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
    
    virtual char *toLuaString() {
      buffer_list buffer;
      buffer.printf("ann.components.base{ name='%s', weights='%s', size=%d }",
		    name.c_str(), weights_name.c_str(), input_size);
      return buffer.to_string(buffer_list::NULL_TERMINATED);
    }
  };
}

#undef MAX_NAME_STR

#endif // ANNCOMPONENT_H
