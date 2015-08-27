/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#include "activation_function_kernels.h"
#include "probabilistic_matrix_component.h"
#include "matrixFloat.h"
#include "sparse_matrixFloat.h"
#include "swap.h"
#include "token_base.h"
#include "token_matrix.h"
#include "table_of_token_codes.h"
#include "token_sparse_matrix.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  ///////////////////////////////////////////
  // ProbabilisticMatrixANNComponent implementation //
  ///////////////////////////////////////////
  
  ProbabilisticMatrixANNComponent::
  ProbabilisticMatrixANNComponent(NormalizationSide side,
                                  const char *name,
                                  const char *weights_name,
                                  unsigned int input_size,
                                  unsigned int output_size,
                                  MatrixFloat *matrix) :
    VirtualMatrixANNComponent(name, weights_name, input_size, output_size),
    weights_mat(matrix),
    needs_weights_normalization(true),
    side(side) {
    setInputContiguousProperty(true);
    if (weights_name == 0) generateDefaultWeightsName("w");
  }
  
  ProbabilisticMatrixANNComponent::~ProbabilisticMatrixANNComponent() {
  }
  
  MatrixFloat *ProbabilisticMatrixANNComponent::
  privateDoForward(MatrixFloat *input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    if (weights_mat.empty()) ERROR_EXIT1(129, "Not built component %s\n",
                                         getName().c_str());
    if (needs_weights_normalization) {
      // compute softmax over weights matrix in order to convert raw weights into
      // linear combination coefficients (sum 1, values in range [0,1])
      if (side == LEFT) {
        april_assert(!T_weights_mat.empty());
        april_assert(!T_normalized_weights_mat.empty());
        Kernels::applySoftmax(T_normalized_weights_mat.get(),
                              T_weights_mat.get());
      }
      else {
        Kernels::applySoftmax(normalized_weights_mat.get(),
                              weights_mat.get());
      }
      // when not training, this flag would be true the first iteration and
      // false in the following
      needs_weights_normalization = during_training;
    }
    //
    unsigned int bunch_size  = input_mat->getDimSize(0);
    // new output to fit the bunch
    MatrixFloat *output_mat;
    int dims[2] = { static_cast<int>(bunch_size),
                    static_cast<int>(getOutputSize()) };
    output_mat = new MatrixFloat(2, dims);
#ifdef USE_CUDA
    output_mat->setUseCuda(use_cuda);
#endif
    if (bunch_size == 1) {
      // vector x matrix product
      matGemv(output_mat,
              CblasNoTrans,
              1.0f, normalized_weights_mat.get(),
              input_mat,
              0.0f);
    } // if bunch_size==1
    else {
      // matrix x matrix product
      // C = \alpha op(A) op(B) + \beta C
      // input * weights = output
      matGemm(output_mat,
              CblasNoTrans,
              CblasTrans,
              1.0f, input_mat, normalized_weights_mat.get(),
              0.0f);
    } // if bunch_size==1 ... else
    return output_mat;
  }
  
  MatrixFloat *ProbabilisticMatrixANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    // new error output to fit the bunch
    unsigned int bunch_size = error_input_mat->getDimSize(0);
    // new output to fit the bunch
    MatrixFloat *error_output_mat;
    int dims[2] = { static_cast<int>(bunch_size),
		    static_cast<int>(getInputSize()) };
    error_output_mat = new MatrixFloat(2, dims);
#ifdef USE_CUDA
    error_output_mat->setUseCuda(use_cuda);
#endif      
    //
    if (bunch_size > 1) {
      // C = alpha * A * B + beta * C
      matGemm(error_output_mat,
              CblasNoTrans, CblasNoTrans,
              1.0f, error_input_mat,
              normalized_weights_mat.get(),
              0.0f);
    }
    else {
      matGemv(error_output_mat,
              CblasTrans,
              1.0f, normalized_weights_mat.get(),
              error_input_mat,
              0.0f);
    }
    return error_output_mat;
  }

  void ProbabilisticMatrixANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    weights_mat->resetSharedCount();
  }

  MatrixFloat *ProbabilisticMatrixANNComponent::
  initializeComputeGradients(const char *name,
                             AprilUtils::LuaTable &grads_mat_dict) {
    weights_mat->addToSharedCount();
    MatrixFloat *grads_mat = grads_mat_dict.opt<MatrixFloat*>(name, 0);
    if (grads_mat == 0) {
      grads_mat = weights_mat->cloneOnlyDims();
      matZeros(grads_mat);
      grads_mat_dict.put<MatrixFloat*>(name, grads_mat);
    }
    else if (!grads_mat->sameDim(weights_mat.get())) {
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
    }
#ifdef USE_CUDA
    grads_mat->setUseCuda(use_cuda);
#endif
    return grads_mat;
  }
  
  void ProbabilisticMatrixANNComponent::
  computeGradients(const char *name,
                   AprilUtils::LuaTable & grads_mat_dict) {
    MatrixFloat *grads_mat = initializeComputeGradients(name, grads_mat_dict);
    MatrixFloat *error_input_mat;
    error_input_mat = getErrorInputMatrix();
    MatrixFloat *input_mat = getInputMatrix();
    // compute gradients respect to normalized weights
    AprilUtils::SharedPtr<MatrixFloat> norm_grads_mat(grads_mat->cloneOnlyDims());
    matGemm(norm_grads_mat.get(),
            CblasTrans, CblasNoTrans,
            1.0f,
            error_input_mat,  // A
            input_mat,        // B
            0.0f);
    int norm_dim;
    // multiply by softmax derivative to obtain non-normalized weight gradients
    if (side == LEFT) {
      april_assert(!T_normalized_weights_mat.empty());
      norm_dim = 0;
      AprilUtils::SharedPtr<MatrixFloat> T_grads_mat(grads_mat->transpose());
      AprilUtils::SharedPtr<MatrixFloat> T_norm_grads_mat(norm_grads_mat->transpose());
      Kernels::applySoftmaxDerivative(T_grads_mat.get(), T_norm_grads_mat.get(),
                                      T_normalized_weights_mat.get());
    }
    else {
      norm_dim = 1;
      Kernels::applySoftmaxDerivative(grads_mat, norm_grads_mat.get(),
                                      normalized_weights_mat.get());
    }
    // normalize the weights, to avoid weights exploding
    AprilUtils::SharedPtr<MatrixFloat> max( matMax(weights_mat.get(), norm_dim) );
    AprilUtils::SharedPtr<MatrixFloat> row;
    for (int i=0; i<weights_mat->getDimSize(norm_dim); ++i) {
      row = weights_mat->select(norm_dim, i);
      matAxpy(row.get(), -1.0f, max.get());
    }
    matClamp(weights_mat.get(), -30.0f, 1.0f);
    // just in case, if gradients have been computed, weights may be changed
    // after, so update the flag
    needs_weights_normalization = true;
  }
  
  ANNComponent *ProbabilisticMatrixANNComponent::clone(AprilUtils::LuaTable &copies) {
    UNUSED_VARIABLE(copies);
    ProbabilisticMatrixANNComponent *component = new
      ProbabilisticMatrixANNComponent(side, getName().c_str(),
                                      getWeightsName().c_str(),
                                      getInputSize(), getOutputSize());
    return component;
  }
  
  void ProbabilisticMatrixANNComponent::build(unsigned int _input_size,
                                              unsigned int _output_size,
                                              AprilUtils::LuaTable &weights_dict,
                                              AprilUtils::LuaTable &components_dict) {
    VirtualMatrixANNComponent::build(_input_size, _output_size,
                                     weights_dict, components_dict);
    //
    if (getInputSize() == 0 || getOutputSize() == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  getName().c_str());
    unsigned int weights_input_size  = getInputSize();
    unsigned int weights_output_size = getOutputSize();
    ////////////////////////////////////////////////////////////////////
    MatrixFloat *w = weights_dict.opt<MatrixFloat*>(getWeightsName(), 0);
    // printf("%s :: %p %p\n", weights_name.c_str(), w, weights_mat);
    if (w != 0) {
      // printf("COPY OF WEIGHTS FROM HASH %s\n", weights_name.c_str());
      weights_mat = w;
      if (!Connections::checkInputOutputSizes(weights_mat.get(),
					      weights_input_size,
					      weights_output_size))
	ERROR_EXIT5(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d and %d, found %d and %d [%s]\n",
		    weights_input_size, weights_output_size,
		    Connections::getInputSize(weights_mat.get()),
		    Connections::getOutputSize(weights_mat.get()),
                    getName().c_str());
    }
    else {
      if (weights_mat.empty()) {
	// printf("NEW OF WEIGHTS %s\n", weights_name.c_str());
	weights_mat = Connections::build(weights_input_size,
                                         weights_output_size);
      }
      // else printf("USING PREVIOUS WEIGHTS %s\n", weights_name.c_str());
      weights_dict.put<MatrixFloat*>(getWeightsName(), weights_mat.get());
    }
    normalized_weights_mat = weights_mat->cloneOnlyDims();
    if (side == LEFT) {
      T_weights_mat            = weights_mat->transpose();
      T_normalized_weights_mat = normalized_weights_mat->transpose();
    }
  }

  void ProbabilisticMatrixANNComponent::copyWeights(AprilUtils::LuaTable &weights_dict) {
    if (!weights_mat) {
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  getName().c_str());
    }
    MatrixFloat *w = weights_dict.opt<MatrixFloat*>(getWeightsName(), 0);
    if (w != 0 && w != weights_mat.get())
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with weights_mat attribute [%s]\n",
		  getWeightsName().c_str(),
		  getName().c_str());
    else if (w == 0) {
      weights_dict.put<MatrixFloat*>(getWeightsName(), weights_mat.get());
    }
  }  

  const char *ProbabilisticMatrixANNComponent::luaCtorName() const {
    return "ann.components.probabilistic_matrix";
  }
  int ProbabilisticMatrixANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["side"]    = (side == LEFT) ? "left" : "right";
    t["name"]    = getName().c_str();
    t["weights"] = getWeightsName().c_str();
    t["input"]   = getInputSize();
    t["output"]  = getOutputSize();
    t["matrix"]  = weights_mat.get();
    t.pushTable(L);
    return 1;
  }
  //////////////////////////////////////////////////////////////////////////
}
