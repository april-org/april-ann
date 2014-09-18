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
#include "aux_hash_table.h" // required for build
#include "connection.h"
#include "disallow_class_methods.h"
#include "error_print.h"
#include "function_interface.h"
#include "hash_table.h"     // required for build
#include "mystring.h"
#include "token_base.h"
#include "matrixFloat.h"
#include "matrixFloatSet.h"
#include "unused_variable.h"
#include "vector.h"

using AprilUtils::hash;    // required for build
using AprilUtils::string;
using AprilUtils::vector;

#ifndef NDEBUG
#define ASSERT_MATRIX(m) do {						\
    april_assert( (m)->getMajorOrder() == CblasColMajor );		\
  } while(0)
// april_assert( (m)->getNumDim() == 2 );
#else
#define ASSERT_MATRIX(m)
#endif

/**
 * @brief Maximum size of automatically generated names.
 *
 * @see Methods ANN::ANNComponent::generateDefaultWeightsName(),
 * ANN::ANNComponent::generateDefaultWeightsName().
 */
#define MAX_NAME_STR 256

/**
 * @brief All ANN components and other stuff is implemented here.
 */
namespace ANN {
  
  unsigned int mult(const int *v, int n);

  /**
   * @brief Virtual class with basic interface and behavior of ANNs.
   *
   * @note It is a Referenced class and needs heap allocation.
   *
   * An ANN component is the basic module which allows to define Artificial
   * Neural Networks (ANNs). Its basic interface allows to compute forward step,
   * back-propagate errors and weight gradients computation.
   
   * Some ANNComponent's can contain parameters or weights which depend upon the
   * final model topology and can be computed by some basic introspection
   * technique. For this purpose, an ANNComponent can be in two states:
   * non-built and built. Build method changes this state and allows the
   * ANNComponent to allocate the necessary resources derived from its
   * input/output connections.
   *
   * Every ANNComponent has a unique name which allows to locate and identify
   * it. Optionally, a non-unique weights_name is given and allows to locate and
   * identify the component parameters. ANNComponents which share the same
   * weights_name will share also the same weight parameters after the build
   * call.
   *
   * ANNComponent's work majorly over Basics::MatrixFloat and
   * Basics::SparseMatrixFloat classes. Both objects can be put together by
   * using Basics::Token instances, and the methods defined here receive and
   * produce always Basics::Token references.
   */
  class ANNComponent : public Functions::FunctionInterface {
    APRIL_DISALLOW_COPY_AND_ASSIGN(ANNComponent);
  public:
    
    /**
     * @brief Constructor with names and input/output sizes.
     *
     * @param name - The name of this instance.
     *
     * @param weights_name - Optional name for the weight parameters.
     *
     * @param input_size - Optional input size of the ANNComponent.
     *
     * @param output_size - Optional output size of the ANNComponent.
     *
     * @note A value of @c weights_name=0 indicates that non parameters are
     * needed by the ANNComponent. A value @c input_size=0 or @c output_size=0
     * indicates that input or output sizes are unknown and tentatively any of
     * them can be dynamic (different patterns can have different sizes) if the
     * ANNComponent supports that.
     */
    ANNComponent(const char *name, const char *weights_name=0,
                 unsigned int input_size=0, unsigned int output_size=0) :
      input_size(input_size), output_size(output_size),
      use_cuda(AprilMath::GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT) {
      if (name) this->name = AprilUtils::string(name);
      else generateDefaultName();
      if (weights_name) this->weights_name = AprilUtils::string(weights_name);
    }
    
    /// Destructor of ANNComponent.
    virtual ~ANNComponent() { }
    
    /// Returns the name of the component.
    const AprilUtils::string &getName() const { return name; }
    /**
     * Returns the name of the component weight parameters.
     *
     * @note The weights_name string can be an empty string (NULL string).
     */
    const AprilUtils::string &getWeightsName() const { return weights_name; }
    /// Indicates if the weights_name string is valid, i.e., it is not empty.
    bool hasWeightsName() const { return !weights_name.empty(); }
    
    /**
     * @brief Sets @c next_name_id=0 and @c next_weights_id=0.
     *
     * ANNComponent implements a basic procedure to produce automatic name
     * and/or weight_name generation. It uses two counters which are increased
     * with every automatically generated name. generateDefaultName() and
     * generateDefaultWeightsName() are the methods available for these
     * purposes.
     */
    static void resetIdCounters() { next_name_id=0; next_weights_id=0; }
    
    /// Returns the built state of the ANNComponent.
    bool getIsBuilt() const { return is_built; }
    
    /**
     * @brief Generates a default name for weight parameters.
     *
     * The weights_name is generated with the given prefix.
     *
     * @note If not prefix or @c prefix=0 is given, a @c "w" will be used by
     * default.
     *
     * @see Method resetIdCounters().
     */
    void generateDefaultWeightsName(const char *prefix=0) {
      char str_id[MAX_NAME_STR+1];
      char default_prefix[2] = "w";
      if (prefix == 0) prefix = default_prefix;
      snprintf(str_id, MAX_NAME_STR, "%s%u", prefix, next_weights_id);
      weights_name = AprilUtils::string(str_id);
      ++next_weights_id;
    }
    
    // FunctionInterface methods
    virtual unsigned int getInputSize() const {
      return input_size;
    }
    virtual unsigned int getOutputSize() const {
      return output_size;
    }
    virtual Basics::Token *calculate(Basics::Token *input) {
      AprilUtils::SharedPtr<Basics::Token> out( this->doForward(input, false) );
      reset();
      Basics::Token *ptr = out.release(); ReleaseRef(ptr);
      return ptr;
    }
    /////////////////////////////////////////////
    
    /**
     * @brief Computes the output size given a vector of dimension sizes.
     *
     * @param input_size - A AprilUtils::vector with a list of input dimension
     * sizes.
     *
     * @param[out] output_size - A AprilUtils::vector reference which will
     * contain the output dimension sizes for the current ANNComponent with the
     * given @c input_size vector.
     *
     * This method is useful to compute output size of convolutional components.
     *
     * @note The given @c output_size reference is cleared before appending the
     * result, i.e., AprilUtils::vector::clear() method is called as first
     * instruction.
     *
     * @note Identity function is implemented by default in the ANNComponent
     * virtual class.
     */
    virtual void precomputeOutputSize(const AprilUtils::vector<unsigned int> &input_size,
				      AprilUtils::vector<unsigned int> &output_size) {
      output_size.clear();
      if (getOutputSize()>0) output_size.push_back(getOutputSize());
      else if (getInputSize() > 0) output_size.push_back(getInputSize());
      else output_size = input_size;
    }
    
    /// Returns a Basics::Token with the last given input.
    virtual Basics::Token *getInput() { return 0; }
    /// Returns a Basics::Token with the last produced output.
    virtual Basics::Token *getOutput() { return 0; }
    /// Returns a Basics::Token with the last given deltas (error input)
    virtual Basics::Token *getErrorInput() { return 0; }
    /// Returns a Basics::Token with the last produced deltas (error output)
    virtual Basics::Token *getErrorOutput() { return 0; }

    /**
     * @brief Computes the forward step of the ANNComponent.
     *
     * @param input - A Basics::Token with the input given to the ANNComponent.
     *
     * @param during_training - A bool which allows to indicate if the
     * forward step is during training phase or not. Some ANNComponent's need
     * this information to modify its behavior, e.g., DropoutANNComponent uses
     * it to implement dropout technique.
     *
     * @return A new Basics::Token reference with the forward step result.
     *
     * This method executes the set of operations required for each block of
     * connections when performing the forward step of the Backpropagation
     * algorithm and returns its output.
     *
     * @note Identity function is implemented by default in the ANNComponent
     * virtual class.
     */
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training) {
      UNUSED_VARIABLE(during_training);
      return input;
    }

    /**
     * @brief Computes the back-propagation of delta errors step.
     *
     * @param input_error - A Basics::Token with the delta errors of the
     * ANNComponent outputs.
     *
     * @return A Basics::Token reference with the delta errors of the
     * ANNComponent inputs (@c output_error).
     *
     * This method back-propagates error derivatives, i.e., delta
     * errors. Receives input error gradients, and returns its output error
     * gradients Basics::Token.
     *
     * @note By default the identity function is implemented by ANNComponent
     * virtual class.
     */
    virtual Basics::Token *doBackprop(Basics::Token *input_error) {
      return input_error;
    }
    
    /**
     * @brief Releases the references of Basics::Token acquired at doForward()
     * and doBackprop() methods.
     *
     * This method receives a counter of the number of times it is sequentially
     * called by optimizers (as conjugate gradient) using the same input/output
     * patterns.
     *
     * @param it - The number of times it has been called sequentially with the
     * same input/output patterns.
     *
     * @note StochasticANNComponent uses the given @c it value to reuse the same
     * random sequence in every doForward(). A value of @c it=0 changes the
     * random sequence.
     */
    virtual void reset(unsigned int it=0) {
      UNUSED_VARIABLE(it);
    }

    /**
     * @brief Computation of gradient of all ANNComponent's is done here.
     *
     * @param[in,out] weight_grads_dict - A Basics::MatrixFloatSet reference where
     * gradient matrices will be stored.
     *
     * This method traverses all the ANNComponent's using the given
     * Basics::MatrixFloatSet. If hasWeightsName() is true, the method
     * computeGradients() will be executed with the shared Basics::MatrixFloat
     * reference (i.e. AprilUtils::SharedPtr) related to the @c weights_name
     * property.
     *
     * @note The @c weight_grads_dict[weights_name] can be an empty reference,
     * in this case, the called method has the responsability of its proper
     * initialization.
     */
    virtual void computeAllGradients(Basics::MatrixFloatSet *weight_grads_dict){
      if (!weights_name.empty()) {
        computeGradients( (*weight_grads_dict)[weights_name].getDense() );
      }
    }
    
    /**
     * @brief Returns a non-built clone of the caller instance.
     *
     * @return A new ANNComponent reference in non-built state.
     *
     * @note The returned ANNComponent is in non-built state and its weight
     * parameters are set to an NULL reference. It is needed to call built
     * method with a proper weight parameters Basics::MatrixFloat.
     */
    virtual ANNComponent *clone() {
      return new ANNComponent(name.c_str(), weights_name.c_str(),
			      input_size, output_size);
    }
    
    /**
     * @brief Sets the @c use_cuda value.
     *
     * @param v - A bool with the desired value for @c use_cuda flag.
     *
     * @note Derived classes which rewrite this method must call its parent
     * method before doing anything.
     */
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
    
    /// Getter for @c use_cuda property.
    bool getUseCuda() const { return use_cuda; }
    
    /**
     * @brief Method which changes ANNComponent state from non-built to built.
     *
     * This method finish the building of ANNComponent's topology and set
     * weight parameter object pointers.
     *
     * @note All derived classes which rewrite this method must call parent
     * method before doing anything.
     *
     * @param _input_size - The input size given to the method. It can be @c
     * _input_size=0 to indicate that it is unknown or don't care.
     *
     * @param _output_size - The output size given to the method. It can be @c
     * _input_size=0 to indicate that it is unknown or don't care.
     *
     * @param[in,out] weights_dict - A pointer to Basics::MatrixFloatSet where
     * weight matrices are stored.
     *
     * @param[out] components_dict - A dictionary of ANNComponent's which are
     * part of the ANN.
     *
     * @note Derived classes must re-implement this method throwing errors if
     * necessary when input/output sizes have unexpected values, and calling to
     * the parent method before doing anything.
     *
     * @note The @c weights_dict param contains weight Basics::MatrixFloat
     * references (i.e. AprilUtils::SharedPtr) indexed by @c weights_name
     * property. The reference can be empty and the derived class is responsible
     * to initialize it properly. If it is not empty, the derived class is
     * responsible to check its size correctness.
     */
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       Basics::MatrixFloatSet *weights_dict,
		       AprilUtils::hash<AprilUtils::string,ANNComponent*> &components_dict) {
      UNUSED_VARIABLE(weights_dict);
      // if (is_built) ERROR_EXIT(128, "Rebuild is forbidden!!!!\n");
      is_built = true;
      ////////////////////////////////////////////////////////////////////
      ANNComponent *&component = components_dict[name];
      if (component != 0 &&
          component != this) ERROR_EXIT1(102, "Non unique component name found: %s\n",
                                         name.c_str());
      else component = this;
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
    
    /// Retrieve matrix weights from ANNComponent's.
    virtual void copyWeights(Basics::MatrixFloatSet *weights_dict) {
      UNUSED_VARIABLE(weights_dict);
    }

    /**
     * @brief Abstract method to retrieve ANNComponent object.
     *
     * @note All derived classes which rewrite this method must call parent
     * method before doing anything.
     */
    virtual void copyComponents(AprilUtils::hash<AprilUtils::string,ANNComponent*> &components_dict) {
      components_dict[name] = this;
    }
    
    /// For debug purposes.
    virtual void debugInfo() {
      fprintf(stderr, "Component '%s' ('%s')  %d inputs   %d outputs\n",
	      name.c_str(),
	      !(weights_name.empty())?weights_name.c_str():"(null)",
	      input_size, output_size);
    }
    
    /**
     * @brief Looks for an ANNComponent given its name.
     *
     * @param name - The name of the ANNComponent your are looking for.
     *
     * @return An ANNComponent pointer identified by @c name, or a NULL pointer
     * if the given name doesn't exists.
     *
     * @note By default, a leaf component only contains itself, but composed
     * components need to check itself name and the name of all the contained
     * components.
     *
     * @note All derived classes which rewrite this method must call the parent
     * method before doing anything.
     */
    virtual ANNComponent *getComponent(AprilUtils::string &name) {
      if (this->name == name) return this;
      return 0;
    }
    
    /**
     * @brief Returns a string with Lua code for its instantiation in non-built state.
     *
     * @return A C string buffer.
     *
     * @note ANNComponent's are instantiated only with the non-trainable
     * parameters. Its trainable weight parameters had to be given in build
     * method.
     */
    virtual char *toLuaString() {
      AprilUtils::buffer_list buffer;
      buffer.printf("ann.components.base{ name='%s', weights='%s', size=%d }",
		    name.c_str(), weights_name.c_str(), input_size);
      return buffer.to_string(AprilUtils::buffer_list::NULL_TERMINATED);
    }

  private:
    /// A flag which indicates if the ANNComponent has been properly built.
    bool is_built;

    /**
     * @brief Generates a default name for the ANNComponent.
     *
     * The name is generated with the given prefix.
     *
     * @note If not prefix or @c prefix=0 is given, a @c "c" will be used by
     * default.
     *
     * @see Method resetIdCounters().
     */
    void generateDefaultName(const char *prefix=0) {
      char default_prefix[2] = "c";
      char str_id[MAX_NAME_STR+1];
      if (prefix == 0) prefix = default_prefix;
      snprintf(str_id, MAX_NAME_STR, "%s%u", prefix, next_name_id);
      name = AprilUtils::string(str_id);
      ++next_name_id;
    }
    
  protected:
    /// The counter for automatic name generation.
    static unsigned int next_name_id;
    /// The counter for automatic weights_name generation.
    static unsigned int next_weights_id;
    /// The name which identifies the ANNComponent.
    AprilUtils::string name;
    /// The name which identifies the ANNComponent weight parameters.
    AprilUtils::string weights_name;
    /// The input size (or domain) of the ANNComponent.
    unsigned int input_size;
    /// The output size (or range) of the ANNComponent.
    unsigned int output_size;
    /// The @c use_cuda flag.
    bool use_cuda;

    /**
     * @brief Computes the gradient of the weight parameters.
     *
     * This method is rewritten only by ANNComponent's which contain trainable
     * weight matrices, and therefore it is needed to compute its gradients.
     *
     * @param weight_grads - A shared reference (i.e. AprilUtils::SharedPtr) to
     * a Basics::MatrixFloat pointer.
     *
     * @note The default implementation in ANNComponent does nothing.
     *
     * @note The given weight_grads reference can be empty, and the derived
     * class is responsible to initialize it properly, or to check the
     * correctness of sizes and dimensions.
     */
    virtual void computeGradients(AprilUtils::SharedPtr<Basics::MatrixFloat> &weight_grads) {
      UNUSED_VARIABLE(weight_grads);
    }
    
  };
} // namespace ANN

#undef MAX_NAME_STR

#endif // ANNCOMPONENT_H
