/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#include "bias_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  BiasANNComponent::BiasANNComponent(unsigned int size,
				     const char *name,
				     const char *weights_name,
                                     MatrixFloat *matrix) :
    VirtualMatrixANNComponent(name, weights_name, size, size),
    bias_vector(matrix) {
    setInputContiguousProperty(true);
    if (weights_name == 0) generateDefaultWeightsName("b");
  }

  BiasANNComponent::~BiasANNComponent() {
  }

  MatrixFloat *BiasANNComponent::privateDoForward(MatrixFloat* input,
                                                  bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (input->getNumDim() < 2)
      ERROR_EXIT2(128, "At 2-dimensional matrix is expected, found %d. "
		  "[%s]", input->getNumDim(), name.c_str());
    if (!bias_vector) ERROR_EXIT1(129, "Not built component %s\n",
                                  name.c_str());
    unsigned int bunch_size = input->getDimSize(0);
    // linear transfer of input to output
    MatrixFloat *output = input->clone();
    // bias
    MatrixFloat *bias_ptr = bias_vector.get();
    if (bunch_size == 1) {
      matAxpy(output, 1.0f, bias_ptr);
    }
    else {
      // addition of bias vector at output
      doAxpyLoop(output_size, 1.0f,
		 bias_ptr->getRawDataAccess(), bias_ptr->getStrideSize(0), 0,
		 output->getRawDataAccess(), output->getStrideSize(1), 0,
		 bunch_size,
		 0, output->getStrideSize(0),
		 use_cuda);
    }
    //
    return output;
  }

  /// In BiasANNComponent this method is a by-pass
  MatrixFloat *BiasANNComponent::privateDoBackprop(MatrixFloat *error_input)
  {
    return error_input;
  }

  void BiasANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    // reset shared counter
    bias_vector->resetSharedCount();
  }
  
  void BiasANNComponent::computeGradients(const char *name,
                                          AprilUtils::LuaTable &weight_grads_dict) {
    // count one use of the vector
    bias_vector->addToSharedCount();
    MatrixFloat *grads_mat = weight_grads_dict.opt<MatrixFloat*>(name, 0);
    if (grads_mat == 0) {
      grads_mat = bias_vector->cloneOnlyDims();
      matZeros(grads_mat);
      weight_grads_dict.put(name, grads_mat);
    }
    else if (!grads_mat->sameDim(bias_vector.get())) {
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
    }
#ifdef USE_CUDA
    grads_mat->setUseCuda(use_cuda);
#endif
    MatrixFloat *error_input_mat = getErrorInputMatrix();
    unsigned int bunch_size = error_input_mat->getDimSize(0);
    // bias update: prev_bias[j] = prev_bias[j] + \sum_b norm_learn_rate * ERROR_INPUT[b,j]
    if (bunch_size == 1) {
      matAxpy(grads_mat, 1.0f, error_input_mat);
    }
    else {
      doAxpyLoop(output_size,
                 1.0f,
                 error_input_mat->getRawDataAccess(),
                 error_input_mat->getStrideSize(1),
                 0,
                 grads_mat->getRawDataAccess(),
                 grads_mat->getStrideSize(0),
                 0,
                 bunch_size,
                 error_input_mat->getStrideSize(0), 0,
                 use_cuda);
    }
  }

  ANNComponent *BiasANNComponent::clone(AprilUtils::LuaTable &copies) {
    UNUSED_VARIABLE(copies);
    BiasANNComponent *component = new BiasANNComponent(input_size,
						       name.c_str(),
						       weights_name.c_str());
    return component;
  }

  void BiasANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       AprilUtils::LuaTable &weights_dict,
			       AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    if (input_size == 0 && output_size == 0) {
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  name.c_str());
    }
    if (input_size == 0) input_size = output_size;
    else if (output_size == 0) output_size= input_size;
    else if (input_size != output_size) {
      ERROR_EXIT1(142, "BiasANNComponent input/output sizes must be equal [%s]\n",
		  name.c_str());
    }
    unsigned int weights_input_size  = 1;
    unsigned int weights_output_size = output_size;
    ////////////////////////////////////////////////////////////////////
    MatrixFloat *w = weights_dict.opt<MatrixFloat*>(weights_name, 0);
    // printf("%s :: %p %p\n", weights_name.c_str(), w, bias_vector);
    if (w != 0) {
      bias_vector = w;
      // printf("COPY OF BIAS FROM HASH %s\n", weights_name.c_str());
      if (!Connections::checkInputOutputSizes(bias_vector.get(),
					      weights_input_size,
					      weights_output_size))
	ERROR_EXIT3(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d inputs and %d outputs. [%s]\n",
		    weights_input_size, weights_output_size,
		    name.c_str());
    }
    else {
      if (bias_vector.empty()) {
	// printf("NEW OF BIAS %s\n", weights_name.c_str());
	bias_vector = Connections::build(weights_input_size,
					 weights_output_size);
      }
      // else printf("USING PREVIOUS BIAS %s\n", weights_name.c_str());
      weights_dict.put(weights_name, bias_vector.get());
    }
  }

  void BiasANNComponent::copyWeights(AprilUtils::LuaTable &weights_dict) {
    if (bias_vector.empty())
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    MatrixFloat *w = weights_dict.opt<MatrixFloat*>(weights_name, 0);
    if (w != 0 && w != bias_vector.get())
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with bias_vector attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (w == 0) {
      weights_dict.put(weights_name, bias_vector.get());
    }
  }

  const char *BiasANNComponent::luaCtorName() const {
    return "ann.components.bias";
  }
  int BiasANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["name"]    = name;
    t["weights"] = weights_name;
    t["size"]    = getOutputSize();
    t["matrix"]  = bias_vector.get();
    t.pushTable(L);
    return 1;
  }
}
