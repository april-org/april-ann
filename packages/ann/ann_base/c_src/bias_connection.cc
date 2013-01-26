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
#include "error_print.h"
#include "bias_connection.h"
#include "maxmin.h"

using april_utils::max;

namespace ANN {

  BiasConnections::BiasConnections(unsigned int bias_size) :
    Connections(bias_size, 1, bias_size) {
  }
  
  bool BiasConnections::checkInputOutputSizes(ActivationUnits *input,
					      ActivationUnits *output) const {
    if (input != 0 && output != 0)
      ERROR_EXIT(124, "Bias connection only could be associated with "
		 "input or output, but not both\n");
    if (input != 0 && total_size != input->numNeurons()) {
      ERROR_PRINT("Incorrect input size!!!\n");
      return false;
    }
    else if (output != 0 && total_size != output->numNeurons()) {
      ERROR_PRINT("Incorrect output size!!!\n");
      return false;
    }
    return true;
  }

  
  // Crea de forma aleatoria el conjunto de pesos con valores en el
  // rango [low, high]
  void BiasConnections::randomizeWeights(MTRand *rnd, float low, float high) {
    double dsup  = high;
    double dinf  = low;
    double rango = dsup - dinf;
#define rnd_weight(value) do {			\
      (value) = rnd->rand(rango)+dinf;	\
    } while (fabs((value)) < weightnearzero);
    
    float *w      = weights->getPPALForReadAndWrite();
    float *prev_w = prev_weights->getPPALForReadAndWrite();
    
    for (unsigned int j=0; j<total_size; ++j) {
      rnd_weight(w[j]);
      prev_w[j] = w[j];
    }
  }
    
  void BiasConnections::randomizeWeightsAtColumn(unsigned int col,
						 MTRand *rnd,
						 float low, float high) {
    double dsup   = high;
    double dinf   = low;
    double rango  = dsup - dinf;
    float *w      = weights->getPPALForReadAndWrite();
    float *prev_w = prev_weights->getPPALForReadAndWrite();
    // solo hay un bias en la columna dada
    rnd_weight(w[col]);
    prev_w[col] = w[col];
  }
#undef rnd_weight
  
  unsigned int BiasConnections::loadWeights(MatrixFloat *data,
					    MatrixFloat *old_data,
					    unsigned int first_weight_pos,
					    unsigned int column_size) {
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*total_size +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size, min_size);
    if (!old_data) old_data = data;
    unsigned int current_w_pos = first_weight_pos;
    float *w                   = weights->getPPALForReadAndWrite();
    float *prev_w              = prev_weights->getPPALForReadAndWrite();
    for (unsigned int j=0; j<total_size; ++j) {
      w[j]           = data->data[current_w_pos];
      prev_w[j]      = old_data->data[current_w_pos];
      current_w_pos += column_size;
    }
    return current_w_pos;
  }

  unsigned int BiasConnections::copyWeightsTo(MatrixFloat *data,
					      MatrixFloat *old_data,
					      unsigned int first_weight_pos,
					      unsigned int column_size) {
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*total_size +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size, min_size);
    
    unsigned int current_w_pos = first_weight_pos;
    const float *w             = weights->getPPALForRead();
    const float *prev_w        = prev_weights->getPPALForRead();
    for (unsigned int j=0; j<total_size; ++j) {
      data->data[current_w_pos]      = w[j];
      old_data->data[current_w_pos]  = prev_w[j];
      current_w_pos                 += column_size;
    }
    return current_w_pos;
  }
    
  // para hacer copias
  Connections *BiasConnections::clone() {
    BiasConnections *conn = new BiasConnections(total_size);
    float *other_w                 = conn->weights->getPPALForReadAndWrite();
    float *other_prev_w            = conn->prev_weights->getPPALForReadAndWrite();
    const float *w                 = weights->getPPALForRead();
    const float *prev_w            = prev_weights->getPPALForRead();
    
    // Podriamos considerar hacer esto con cblas... aceleraria mucho.
    for (unsigned int i = 0; i < total_size; i++) {
      other_prev_w[i] = prev_w[i];
      other_w[i]      = w[i];
    }
    return conn;
  }
}
