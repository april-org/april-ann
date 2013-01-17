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
#include <cstring>

#include "bias_connection.h"
#include "all_all_connection.h"
#include "dot_product_action.h"
#include "forward_bias_action.h"
#include "activations_action.h"
#include "constString.h"
#include "error_print.h"
#include "all_all_mlp.h"
#include "utilMatrixFloat.h"
#include "ignore_result.h"

namespace ANN {

  // Creates a null description
  AllAllMLP::AllAllMLP(ANNConfiguration conf) : MLP(conf) {
    description = 0;
  }

  AllAllMLP::~AllAllMLP() {
    delete[] description;
  }

  // Generates the actions given a description
  void AllAllMLP::generateActionsAllAll(const char *description) {

    vector<ActivationFunction*> activation_functions;
    delete[] this->description;
    this->description = copystr(description);
    // Description example: "20 inputs 30 logistic 10 softmax"  
    unsigned int num_neurons;
    constString token, descrip(description);

    while (descrip.extract_unsigned_int(&num_neurons) &&
	   (token = descrip.extract_token())) {
      // We always use the Real activations type
      ActivationUnits *act = new
	RealActivationUnits(num_neurons,
			    conf,
			    (activations.size() == 0 ? false : true));
      activation_functions.push_back(getActivationFunctionByTypeString(token));
      if (activations.size() == 0 && token != "inputs")
	ERROR_EXIT(256, "The first activation must be 'inputs'\n");
      registerActivationUnits(act);
    }
    if (activations.size() < 2)
      ERROR_EXIT(128, "Impossible to generate a zero layer AllAllMLP\n");

    // We register the input and the output layers of the network
    registerInput(activations[0]);
    registerOutput(activations.back());
  
    // we need three actions for each layer:
    //   1) bias action
    //   2) dot product action (executes all dot products in the layer)
    //   3) activation action [optional]
    for (unsigned int i = 0; i < activations.size() - 1; i++) {
      // 1) bias
      Connections *bias = new BiasConnections(activations[i+1]->numNeurons());
      Action *action    = new ForwardBiasAction(conf, activations[i+1], bias);
      registerConnections(bias);
      registerAction(action);
      // 2) dot products
      Connections *connections=new
	AllAllConnections(activations[i]->numNeurons(),
			  activations[i+1]->numNeurons());
      action = new DotProductAction(conf,
				    activations[i],
				    activations[i+1],
				    connections);
      registerConnections(connections);
      registerAction(action);
      // 3) activation [optional]
      if (activation_functions[i+1] != 0) {
	Action *action = new ActivationsAction(conf,
					       activations[i+1],
					       activation_functions[i+1]);
	registerAction(action);
      }
    }
  }

  void AllAllMLP::randomizeWeights(MTRand *rnd, float low, float high) {
    // step +=2 because connections are stored in groups of two: bias and the
    // rest of weights
    for (unsigned int i=0; i<connections.size(); i += 2) {
      const unsigned int sz = connections[i]->size();
      for (unsigned int k=0; k<sz; ++k) {
	connections[i]->randomizeWeightsAtColumn(k, rnd, low, high);
	connections[i+1]->randomizeWeightsAtColumn(k, rnd, low, high);
      }
    }
  }
  
  // Generates a random network
  void AllAllMLP::generateAllAll(const char *str, MTRand *rnd,
				 float low, float high) {
    generateActionsAllAll(str);
    randomizeWeights(rnd, low, high);
  }
  
  // Generates a network and initializes it with the given weights
  void AllAllMLP::generateAllAll(const char *str,
				 MatrixFloat *weights_mat,
				 MatrixFloat *old_weights_mat) {
    generateActionsAllAll(str);
    unsigned int pos = 0;
    // step +=2 because connections are stored in groups of two: bias and the
    // rest of weights
    for (unsigned int i=0,k=0; i<connections.size(); i+=2,++k) {
      unsigned int colsize = activations[k]->numNeurons()+1;
      // ATTENTION: The loadWeights function returns the next pos value
      connections[i]->loadWeights(weights_mat, old_weights_mat, pos,
				  colsize);
      pos = connections[i+1]->loadWeights(weights_mat, old_weights_mat, pos+1,
					  colsize) - 1;
    }
  }
  
  // Copies the weights of the given matrices
  void AllAllMLP::copyWeightsToMatrix(MatrixFloat **weights_mat,
				      MatrixFloat **old_weights_mat) {
    unsigned int numw = getNumberOfWeights();
    *weights_mat     = new MatrixFloat(1, static_cast<int>(numw));
    *old_weights_mat = new MatrixFloat(1, static_cast<int>(numw));
    
    unsigned int pos = 0;
    // step +=2 because connections are stored in groups of two: bias and the
    // rest of weights
    for (unsigned int i=0,k=0; i<connections.size(); i+=2,++k) {
      unsigned int colsize = activations[k]->numNeurons()+1;
      // ATTENTION: The copyWeightsTo function returns the next pos value
      
      // We copy the bias...
      connections[i]->copyWeightsTo(*weights_mat, 
				    *old_weights_mat,
				    pos,
				    colsize);
      // ...and the other weights
      pos = connections[i+1]->copyWeightsTo(*weights_mat, 
					    *old_weights_mat,
					    // empezamos en +1 porque el primero
					    // es un bias
					    pos+1,
					    colsize) - 1;
      // We substract 1 so it points to the next bias
    }
    
  }

  AllAllMLP *AllAllMLP::clone() {
    AllAllMLP *copy = new AllAllMLP(getConf());
    copy->setErrorFunction(error_func->clone());
    cloneTopologyTo(copy);
    copy->learning_rate  = learning_rate;
    copy->momentum       = momentum;
    copy->weight_decay   = weight_decay;
    copy->c_weight_decay = c_weight_decay;
    copy->description    = copystr(description);
    return copy;
  }

  unsigned int AllAllMLP::getNumberOfWeights() {
    unsigned int numw = 0;
    for (unsigned int i=0; i<connections.size(); ++i)
      numw += connections[i]->size();
    return numw;
}
  
  void AllAllMLP::saveModel(const char *filename) {
    MatrixFloat *weights_mat;
    MatrixFloat *old_weights_mat;
    
    copyWeightsToMatrix(&weights_mat, &old_weights_mat);
    
    char *weights_mat_buffer;
    char *old_weights_mat_buffer;
    saveMatrixFloatToString(weights_mat,     &weights_mat_buffer,     false);
    saveMatrixFloatToString(old_weights_mat, &old_weights_mat_buffer, false);
    
    FILE *f = fopen(filename, "w");
    fprintf(f, "%s\n%s\n%s", description, weights_mat_buffer,
            old_weights_mat_buffer);
    fclose(f);
    
    delete   weights_mat;
    delete[] weights_mat_buffer;
    delete   old_weights_mat;
    delete[] old_weights_mat_buffer;
  }
  
  void AllAllMLP::loadModel(const char *filename) {  
    FILE *f = fopen(filename, "r");
    if (f == 0) {
      ERROR_PRINT1("File '%s' not found\n", filename);
      exit(128);
    }
    
    fseek(f, 0L, SEEK_END);
    unsigned long int sz = ftell(f);
    fseek(f, 0L, SEEK_SET);
    char *buffer = new char[sz+1];
    IGNORE_RESULT(fread(buffer, sizeof(char), sz, f));
    fclose(f);
    
    constString buffer_string(buffer, sz);
    constString description_line  = buffer_string.extract_line();
    char *description_line_string = description_line.newString();
    MatrixFloat *weights_mat;
    MatrixFloat *old_weights_mat;
    weights_mat     = readMatrixFloatFromStream<constString>(buffer_string);
    old_weights_mat = readMatrixFloatFromStream<constString>(buffer_string);
    generateAllAll(description_line_string, weights_mat, old_weights_mat);
    
    delete[] description_line_string;
    delete   weights_mat;
    delete   old_weights_mat;
  }

  Connections *AllAllMLP::getLayerConnections(unsigned int layer)  {
    if (layer >= connections.size()) {
      ERROR_PRINT("Incorrect layer number!!!");
      exit(128);
    }
    return connections[layer];
  }
}
