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
#include "activation_function_kernels.h"
#include "cblas_headers.h"
#include "cmath_overloads.h"
#include "logistic_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  LogisticActfANNComponent::LogisticActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LogisticActfANNComponent::~LogisticActfANNComponent() { }

  void LogisticActfANNComponent::applyActivation(MatrixFloat *input_units,
						 MatrixFloat *output_units) {
    Kernels::applyLogistic(output_units, input_units);
  }
  
  void LogisticActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
						     MatrixFloat *output_units,
						     MatrixFloat *input_errors,
						     MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    Kernels::applyLogisticDerivative(output_errors, output_units);
    matCmul(output_errors, input_errors);
  }

  ANNComponent *LogisticActfANNComponent::clone() {
    LogisticActfANNComponent *obj = new LogisticActfANNComponent(name.c_str());
    return obj;
  }

  char *LogisticActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.logistic{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

  ///////////////////////////////////////////////////////////////////////

  SparseLogisticActfANNComponent::
  SparseLogisticActfANNComponent(const char *name,
                                 float beta, float target_avg_act) :
    LogisticActfANNComponent(name) {
    this->beta = beta; this->target_avg_act = target_avg_act;
  } 
  
  SparseLogisticActfANNComponent::~SparseLogisticActfANNComponent() {
  }

  ANNComponent *SparseLogisticActfANNComponent::clone() {
    SparseLogisticActfANNComponent *obj = new
      SparseLogisticActfANNComponent(name.c_str(), beta, target_avg_act);
    return obj;
  }

  char *SparseLogisticActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.sparse_logistic{ "
                  "name='%s', penalty=%f, sparsity=%f  }",
                  name.c_str(), this->beta, this->target_avg_act);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
 
  void SparseLogisticActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
                                                           MatrixFloat *output_units,
                                                           MatrixFloat *input_errors,
                                                           MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    
    const float INV_DIM0 = 1.0f/static_cast<float>(output_units->getDimSize(0));
    const float EPS = AprilMath::Limits<float>::epsilon();
    // Computing average activations
    MatrixFloat *current_avg = matSum(output_units, 0);
    matScal(current_avg, INV_DIM0);
     
    // -p/\hat{p}
    MatrixFloat *sparse_gradients = current_avg->clone();
    matClamp(sparse_gradients, EPS, 1.0f);
    matDiv(sparse_gradients, -target_avg_act);
    
    // (1-p)/(1-\hat{p})
    MatrixFloat *aux = current_avg; // reusing current_avg
    matComplement(aux);
    matClamp(aux, EPS, 1.0f);
    matDiv(aux, (1.0f-target_avg_act));
    
    // -p/\hat{p} + (1-p)/(1-\hat{p} = sparse_gradients + aux
    matAxpy(sparse_gradients, 1.0f, aux);
    
    // FIXME: Do we need to normalize by the bunch size?
    // matScal(sparse_gradients, INV_DIM0);

    // Unfold
    MatrixFloat *sparse_errors = input_errors->clone();
    MatrixFloat *row = 0; 
    for (int i = 0; i < output_units->getDimSize(0); ++i) {
      row = sparse_errors->select(0, i, row);
      matAxpy(row , beta, sparse_gradients);
    }
    delete row;

    Kernels::applyLogisticDerivative(output_errors, output_units);
    matCmul(output_errors, sparse_errors);
   
    delete aux;
    delete sparse_gradients;
    delete sparse_errors;
  }
}
