/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#include "unused_variable.h"
#include "swap.h"
#include "dot_product_component.h"
#include "wrapper.h"
#include "matrixFloat.h"
#include "sparse_matrixFloat.h"
#include "token_base.h"
#include "token_matrix.h"
#include "token_sparse_matrix.h"
#include "table_of_token_codes.h"

using namespace basics;
using namespace april_utils;
using namespace april_math;

using april_utils::swap;

namespace ANN {

  ///////////////////////////////////////////
  // DotProductANNComponent implementation //
  ///////////////////////////////////////////
  
  DotProductANNComponent::DotProductANNComponent(const char *name,
						 const char *weights_name,
						 unsigned int input_size,
						 unsigned int output_size,
						 bool transpose_weights) :
    MatrixInputSwitchANNComponent(name, weights_name, input_size, output_size),
    weights_matrix(0) {
    setInputContiguousProperty(true);
    if (weights_name == 0) generateDefaultWeightsName("w");
    this->transpose_weights = (transpose_weights) ? CblasTrans : CblasNoTrans;
  }
  
  DotProductANNComponent::~DotProductANNComponent() {
    if (weights_matrix) DecRef(weights_matrix);
  }
  
  MatrixFloat *DotProductANNComponent::
  privateDoDenseForward(MatrixFloat *input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (input_mat->getNumDim() < 2)
      ERROR_EXIT2(128, "At 2-dimensional matrix is expected, found %d. "
		  "[%s]", input_mat->getNumDim(), name.c_str());
    if (weights_matrix == 0) ERROR_EXIT1(129, "Not built component %s\n",
                                         getName().c_str());
    MatrixFloat *weights_mat = weights_matrix;
    unsigned int bunch_size  = input_mat->getDimSize(0);
    // new output to fit the bunch
    MatrixFloat *output_mat;
    int dims[2] = { static_cast<int>(bunch_size),
                    static_cast<int>(getOutputSize()) };
    output_mat = new MatrixFloat(2, dims, CblasColMajor);
#ifdef USE_CUDA
    output_mat->setUseCuda(use_cuda);
#endif
    if (bunch_size == 1) {
      // vector x matrix product
      output_mat->gemv(transpose_weights,
                       1.0f, weights_mat,
                       input_mat,
                       0.0f);
    } // if bunch_size==1
    else {
      // matrix x matrix product
      // C = \alpha op(A) op(B) + \beta C
      // input * weights = output
      output_mat->gemm(CblasNoTrans,
                       NEGATE_CBLAS_TRANSPOSE(transpose_weights),
                       1.0f, input_mat, weights_mat,
                       0.0f);
    } // if bunch_size==1 ... else
    return output_mat;
  }
  
  MatrixFloat *DotProductANNComponent::
  privateDoSparseForward(SparseMatrixFloat *input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    unsigned int bunch_size = input_mat->getDimSize(0);
    MatrixFloat *weights_mat = weights_matrix;
    // new output to fit the bunch
    MatrixFloat *output_mat;
    int dims[2] = {static_cast<int>(bunch_size),
                   static_cast<int>(getOutputSize())};
    output_mat = new MatrixFloat(2, dims, CblasColMajor);
#ifdef USE_CUDA
    output_mat->setUseCuda(use_cuda);
#endif
    output_mat->sparseMM(CblasNoTrans,
                         NEGATE_CBLAS_TRANSPOSE(transpose_weights),
                         CblasNoTrans,
                         1.0f, input_mat, weights_mat,
                         0.0f);
    return output_mat;
  }
  
  MatrixFloat *DotProductANNComponent::
  privateDoDenseBackprop(MatrixFloat *error_input_mat) {
    // new error output to fit the bunch
    unsigned int bunch_size = error_input_mat->getDimSize(0);
    // new output to fit the bunch
    MatrixFloat *error_output_mat;
    int dims[2] = { static_cast<int>(bunch_size),
		    static_cast<int>(getInputSize()) };
    error_output_mat = new MatrixFloat(2, dims, CblasColMajor);
#ifdef USE_CUDA
    error_output_mat->setUseCuda(use_cuda);
#endif      
    //
    MatrixFloat *weights_mat = weights_matrix;
    if (bunch_size > 1) {
      // C = alpha * A * B + beta * C
      error_output_mat->gemm(CblasNoTrans, transpose_weights,
			     1.0f, error_input_mat,
			     weights_mat,
			     0.0f);
    }
    else {
      error_output_mat->gemv(NEGATE_CBLAS_TRANSPOSE(transpose_weights),
			     1.0f, weights_mat,
			     error_input_mat,
			     0.0f);
    }
    return error_output_mat;
  }

  SparseMatrixFloat *DotProductANNComponent::
  privateDoSparseBackprop(MatrixFloat *error_input_mat) {
    UNUSED_VARIABLE(error_input_mat);
    // If input is sparse, the component needs to be an input of the ANN,
    // therefore the input is probably SO LARGE, and computing the backprop
    // will lead in HIGH computational cost ;) Because of this, the components
    // returns a NULL gradient pointer
    return 0;
  }
  
  void DotProductANNComponent::privateDenseReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    weights_matrix->resetSharedCount();
  }

  void DotProductANNComponent::privateSparseReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    weights_matrix->resetSharedCount();
  }

  void DotProductANNComponent::
  initializeComputeGradients(MatrixFloat*& grads_mat) {
    weights_matrix->addToSharedCount();
    if (grads_mat == 0) {
      grads_mat = weights_matrix->cloneOnlyDims();
      grads_mat->zeros();
      IncRef(grads_mat);
    }
    else if (!grads_mat->sameDim(weights_matrix))
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
  }
  
  void DotProductANNComponent::
  privateDenseComputeGradients(MatrixFloat*& grads_mat) {
    initializeComputeGradients(grads_mat);
    MatrixFloat *error_input_mat;
    error_input_mat = getErrorInputMatrix();
    unsigned int bunch_size = error_input_mat->getDimSize(0);
    MatrixFloat *input_mat = getInputMatrix();
    if (bunch_size > 1) {
      grads_mat->gemm(CblasTrans, CblasNoTrans,
                      1.0f,
                      (transpose_weights == CblasNoTrans)?error_input_mat:input_mat, // A
                      (transpose_weights == CblasNoTrans)?input_mat:error_input_mat, // B
                      1.0f);
    } // if bunch_size > 1 ... else
    else {
      grads_mat->ger(1.0f,
                     (transpose_weights == CblasNoTrans)?error_input_mat:input_mat,
                     (transpose_weights == CblasNoTrans)?input_mat:error_input_mat);
    } // if bunch_size > 1 ... else
  }
  
  void DotProductANNComponent::
  privateSparseComputeGradients(MatrixFloat*& grads_mat) {
    initializeComputeGradients(grads_mat);
    MatrixFloat *error_input_mat;
    error_input_mat = getErrorInputMatrix();
    SparseMatrixFloat *input_mat;
    input_mat = getSparseInputMatrix();
    if (transpose_weights == CblasNoTrans)
      grads_mat->sparseMM(CblasTrans,
                          CblasNoTrans,
                          CblasTrans,
                          1.0f,
                          input_mat,
                          error_input_mat,
                          1.0f);
    else
      grads_mat->sparseMM(CblasTrans,
                          CblasNoTrans,
                          CblasNoTrans,
                          1.0f,
                          input_mat,
                          error_input_mat,
                          1.0f);
  }
  
  ANNComponent *DotProductANNComponent::clone() {
    DotProductANNComponent *component = new
      DotProductANNComponent(getName().c_str(), getWeightsName().c_str(),
			     getInputSize(), getOutputSize(),
			     (transpose_weights == CblasTrans));
    return component;
  }
  
  void DotProductANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     MatrixFloatSet *weights_dict,
				     hash<string,ANNComponent*> &components_dict) {
    MatrixInputSwitchANNComponent::build(_input_size, _output_size,
                                         weights_dict, components_dict);
    //
    if (getInputSize() == 0 || getOutputSize() == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  getName().c_str());
    unsigned int weights_input_size  = getInputSize();
    unsigned int weights_output_size = getOutputSize();
    ////////////////////////////////////////////////////////////////////
    if (transpose_weights == CblasTrans)
      swap(weights_input_size, weights_output_size);
    MatrixFloat *&w = (*weights_dict)[getWeightsName()];
    // printf("%s :: %p %p\n", weights_name.c_str(), w, weights_matrix);
    if (w != 0) {
      // printf("COPY OF WEIGHTS FROM HASH %s\n", weights_name.c_str());
      AssignRef(weights_matrix, w);
      if (!Connections::checkInputOutputSizes(weights_matrix,
					      weights_input_size,
					      weights_output_size))
	ERROR_EXIT5(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d and %d, found %d and %d [%s]\n",
		    weights_input_size, weights_output_size,
		    Connections::getInputSize(weights_matrix),
		    Connections::getOutputSize(weights_matrix),
                    getName().c_str());
    }
    else {
      if (weights_matrix == 0) {
	// printf("NEW OF WEIGHTS %s\n", weights_name.c_str());
	weights_matrix = Connections::build(weights_input_size,
					    weights_output_size);
	IncRef(weights_matrix);
      }
      // else printf("USING PREVIOUS WEIGHTS %s\n", weights_name.c_str());
      w = weights_matrix;
      IncRef(w);
    }
  }

  void DotProductANNComponent::copyWeights(MatrixFloatSet *weights_dict) {
    if (weights_matrix == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  getName().c_str());
    MatrixFloat *&w = (*weights_dict)[getWeightsName()];
    if (w != 0 && w != weights_matrix)
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with weights_matrix attribute [%s]\n",
		  getWeightsName().c_str(),
		  getName().c_str());
    else if (w == 0) {
      w = weights_matrix;
      IncRef(w);
    }
  }  

  char *DotProductANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.dot_product{ name='%s',weights='%s',"
		  "input=%d,output=%d,transpose=%s }",
		  getName().c_str(), getWeightsName().c_str(),
		  getInputSize(), getOutputSize(),
		  (transpose_weights==CblasTrans)?"true":"false");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  //////////////////////////////////////////////////////////////////////////
}
