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
#include "logistic_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
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

  SparseLogisticActfANNComponent::SparseLogisticActfANNComponent(const char *name,
        float beta, float avg_act):LogisticActfANNComponent(name) {
      this->beta = beta; this->avg_act = avg_act; } 

  ANNComponent *SparseLogisticActfANNComponent::clone() {
    SparseLogisticActfANNComponent *obj = new SparseLogisticActfANNComponent(name.c_str(), beta, avg_act);
    return obj;
  }

  char *SparseLogisticActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.sparse_logistic{ name='%s', beta=%f, rho=%f  }", name.c_str(), this->beta, this->avg_act);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
 
  void SparseLogisticActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
						     MatrixFloat *output_units,
						     MatrixFloat *input_errors,
						     MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    Kernels::applyLogisticDerivative(output_errors, output_units);

    float eps = 1e-6;
    // Computing average activations
    MatrixFloat *current_avg = matSum(output_units, 0);
    matScal(current_avg, 1.0f/static_cast<float>(output_units->getDimSize(0)));
     
    // -p/\hat{p} + (1-p)/(1-\hat{p}
    // -p/\hat{p}
    
    MatrixFloat *sparse_errors = current_avg->clone();
    matScalarAdd(sparse_errors, eps);
    matDiv(sparse_errors, -avg_act);
    
    // (1-p)/(1-\hat{p})
    MatrixFloat *aux = current_avg->clone();
    matComplement(aux);
    matScalarAdd(aux, eps);
    matDiv(aux, (1.0f-avg_act));
    matAddition(sparse_errors, aux, sparse_errors);

    matAbs(sparse_errors, sparse_errors);
    // Normalize the error
    matScal(sparse_errors, 1.0f/output_units->getDimSize(0));

    //Unfold
    MatrixFloat *row = 0; 
    for (int i = 0; i < output_units->getDimSize(0); ++i) {
      row = output_errors->select(0,i, row);
      matAxpy(row , beta, sparse_errors);
    }
  
    delete row;
    matCmul(output_errors, input_errors);
   
    delete aux;
    delete sparse_errors;
  }
}
