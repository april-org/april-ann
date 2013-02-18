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
#include "maxmin.h"
#include "all_all_connection.h"

using april_utils::max;

namespace ANN {

  AllAllConnections::AllAllConnections(unsigned int num_inputs,
				       unsigned int num_outputs) :
    Connections(num_inputs*num_outputs, num_inputs, num_outputs) {
  }
  
  bool AllAllConnections::checkInputOutputSizes(ActivationUnits *input,
						ActivationUnits *output) const {
    // TODO: comprobar error input==0 y output==0
    if (num_inputs != input->numNeurons()) {
      ERROR_PRINT("Incorrect input size!!!\n");
      return false;
    }
    if (num_outputs != output->numNeurons()) {
      ERROR_PRINT("Incorrect output size!!!\n");
      return false;
    }
    return true;
  }

  
  // Crea de forma aleatoria el conjunto de pesos con valores en el
  // rango [low*sqrt(fan_in), high*sqrt(fan_in)]
  void AllAllConnections::randomizeWeights(MTRand *rnd, float low, float high,
					   bool use_fanin) {
    double dinf = low;
    double dsup = high;
    if (use_fanin) {
      double inv_sqrt_fan_in = 1.0/sqrt(fanin);
      dinf *= inv_sqrt_fan_in;
      dsup *= inv_sqrt_fan_in;
    }
    // assert to avoid nearzero weights
    assert(fabs(dinf) > weightnearzero);
    assert(fabs(dsup) > weightnearzero);
    double range  = dsup - dinf;
    float *w      = weights->getPPALForReadAndWrite();
    float *prev_w = prev_weights->getPPALForReadAndWrite();
    for (unsigned int j=0; j<num_outputs; ++j) {
      unsigned int k = j;
      for (unsigned int i=0; i<num_inputs; ++i) {
	rnd_weight(rnd, w[k], dinf, range, weightnearzero);
	prev_w[k] = w[k];
	k += num_outputs;
      }
    }
  }
    
  void AllAllConnections::randomizeWeightsAtColumn(unsigned int col,
						   MTRand *rnd,
						   float low, float high,
						   bool use_fanin) {
    double dinf = low;
    double dsup = high;
    if (use_fanin) {
      double inv_sqrt_fan_in = 1.0/sqrt(fanin);
      dinf *= inv_sqrt_fan_in;
      dsup *= inv_sqrt_fan_in;
    }
    // assert to avoid nearzero weights
    assert(fabs(dinf) > weightnearzero);
    assert(fabs(dsup) > weightnearzero);
    double range  = dsup - dinf;
    float *w      = weights->getPPALForReadAndWrite();
    float *prev_w = prev_weights->getPPALForReadAndWrite();
    unsigned int k = col;
    for (unsigned int i=0; i<num_inputs; ++i) {
      rnd_weight(rnd, w[k], dinf, range, weightnearzero);
      prev_w[k] = w[k];
      k += num_outputs;
    }
  }
  
  unsigned int AllAllConnections::loadWeights(MatrixFloat *data,
					      MatrixFloat *old_data,
					      unsigned int first_weight_pos,
					      unsigned int column_size) {
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*num_outputs +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size, min_size);
    if (!old_data) old_data = data;
    unsigned int current_w_pos = first_weight_pos;
    float *w                   = weights->getPPALForReadAndWrite();
    float *prev_w              = prev_weights->getPPALForReadAndWrite();
    for (unsigned int j=0; j<num_outputs; ++j) {
      unsigned int k = j;
      for (unsigned int i=0; i<num_inputs; ++i) {
	w[k]      = data->data[current_w_pos+i];
	prev_w[k] = old_data->data[current_w_pos+i];
	k += num_outputs;
      }
      current_w_pos += column_size;
    }
    return current_w_pos;
  }

  unsigned int AllAllConnections::copyWeightsTo(MatrixFloat *data,
						MatrixFloat *old_data,
						unsigned int first_weight_pos,
						unsigned int column_size) {
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*num_outputs +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size, min_size);

    unsigned int current_w_pos = first_weight_pos;
    const float *w             = weights->getPPALForRead();
    const float *prev_w        = prev_weights->getPPALForRead();
      
    for (unsigned int j=0; j<num_outputs; ++j) {
      unsigned int k = j;
      for (unsigned int i=0; i<num_inputs; ++i) {
	data->data[current_w_pos+i]     = w[k];
	old_data->data[current_w_pos+i] = prev_w[k];
	k += num_outputs;
      }
      current_w_pos += column_size;
    }
    return current_w_pos;
  }
    
  // para hacer copias
  Connections *AllAllConnections::clone() {
    AllAllConnections *conn = new AllAllConnections(num_inputs, num_outputs);
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

  void AllAllConnections::endUpdate() {
    Connections::endUpdate();
    float *w             = weights->getPPALForReadAndWrite();
    float *prev_w        = prev_weights->getPPALForReadAndWrite();
    for (unsigned int j=0; j<num_outputs; ++j) {
      unsigned int k = j;
      float squared_length = 0.0f;
      for (unsigned int i=0; i<num_inputs; ++i) {
	squared_length += w[k]*w[k];
	k += num_outputs;
      }
      if (squared_length > 15.0f) {
	float ratio    = sqrtf(15.0f/squared_length);
	unsigned int k = j;
	for (unsigned int i=0; i<num_inputs; ++i) {
	  w[k]      *= ratio;
	  prev_w[k] *= ratio;
	  k += num_outputs;
	}
      }
    }    
  }
}
