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
#include "swap.h"
#include "connection.h"
#include "check_floats.h"

namespace ANN {
  const double Connections::weightnearzero = 1e-7;
  
  Connections::Connections(unsigned int total_size,
			   unsigned int num_inputs, unsigned int num_outputs) :
    Referenced(),
    weights(0), prev_weights(0),
    total_size(total_size),
    num_inputs(num_inputs), num_outputs(num_outputs),
    num_references(0), update_weights_calls(0),
    fanin(num_inputs) {
    weights      = new FloatGPUMirroredMemoryBlock(total_size);
    prev_weights = new FloatGPUMirroredMemoryBlock(total_size);
    if (weights == 0 || prev_weights == 0)
      ERROR_EXIT(130, "Impossible to allocate memory\n");
  }

  Connections::~Connections() {
    delete weights;
    delete prev_weights;
  }

  bool AllAllConnections::checkInputOutputSizes(unsigned int input_size,
						unsigned int output_size) const {
    // TODO: comprobar error input==0 y output==0
    if (num_inputs != input_size) {
      ERROR_PRINT("Incorrect input size!!!\n");
      return false;
    }
    if (num_outputs != output_size) {
      ERROR_PRINT("Incorrect output size!!!\n");
      return false;
    }
    return true;
  }

  void Connections::countReference() {
    ++num_references;
  }
    
  unsigned int Connections::getNumReferences() const {
    return num_references;
  }
    
  void Connections::beginUpdate() {
    ++update_weights_calls;
  }
    
  bool Connections::endUpdate() {
    // if it is the last call
    if (update_weights_calls == num_references) {
      // Swap(w, prev_w)
      april_utils::swap(weights, prev_weights);
      update_weights_calls = 0;
      return true;
    }
    return false;
  }
    
  bool Connections::isFirstUpdateCall() {
    return update_weights_calls == 1;
  }

  void Connections::
  computeMomentumOnPrevVector(float momentum, bool use_cuda) {
    // momentum learning rule
    // prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
    //
    // but this method computes: first the complementary with saxpy:
    // prev_w[i,j] = prev_w[i,j] - 1.0f * w[i,j]
    doSaxpy(total_size,
	    -1.0f,
	    weights, 0, 1,
	    prev_weights, 0, 1,
	    use_cuda);
    // second apply momentum with sscal:
    // prev_w[i,j] = -momentum * prev_w[i,j] = -momentum*(prev_w[i,j] - w[i,j])
    doSscal(total_size,
	    -momentum,
	    prev_weights, 0, 1,
	    use_cuda);
  }
  
  void Connections::
  computeWeightDecayOnPrevVector(float c_weight_decay, bool use_cuda) {
    // applies weight decay
    // prev_w[i,j] = c_weight_decay * w[i,j] + prev_w[i,j]
    //
    doSaxpy(total_size,
	    c_weight_decay,
	    weights, 0, 1,
	    prev_weights, 0, 1,
	    use_cuda);
  }

  unsigned int Connections::size() const {
    return total_size;
  }
    
  void Connections::copyToPrevVector(bool use_cuda) {
    doScopy(total_size,
	    weights, 0, 1,
	    prev_weights, 0, 1,
	    use_cuda);
  }
  
  void Connections::pruneSubnormalAndCheckNormal() {
    float *w = weights->getPPALForReadAndWrite();
    if (!april_utils::check_floats(w, total_size)) {
      assert("No finite numbers at weights matrix!!!" && false);
      ERROR_EXIT(128, "No finite numbers at weights matrix!!!\n");
    }
  }

  FloatGPUMirroredMemoryBlock *Connections::getPtr() {
    return weights;
  }

  FloatGPUMirroredMemoryBlock *Connections::getPrevPtr() {
    return prev_weights;
  }

  // Crea de forma aleatoria el conjunto de pesos con valores en el
  // rango [low*sqrt(fan_in), high*sqrt(fan_in)]
  void Connections::randomizeWeights(MTRand *rnd, float low, float high) {
    double dinf = low;
    double dsup = high;
    /*
      if (use_fanin) {
      double inv_sqrt_fan_in = 1.0/sqrt(fanin);
      dinf *= inv_sqrt_fan_in;
      dsup *= inv_sqrt_fan_in;
      }
    */
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
    
  void Connections::randomizeWeightsAtColumn(unsigned int col,
					     MTRand *rnd,
					     float low, float high) {
    double dinf = low;
    double dsup = high;
    /*
      if (use_fanin) {
      double inv_sqrt_fan_in = 1.0/sqrt(fanin);
      dinf *= inv_sqrt_fan_in;
      dsup *= inv_sqrt_fan_in;
      }
    */
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
  
  unsigned int Connections::loadWeights(MatrixFloat *data,
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

  unsigned int Connections::copyWeightsTo(MatrixFloat *data,
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
  Connections *Connections::clone() {
    Connections *conn	      = new Connections(num_inputs, num_outputs);
    float	*other_w      = conn->weights->getPPALForReadAndWrite();
    float	*other_prev_w = conn->prev_weights->getPPALForReadAndWrite();
    const float *w            = weights->getPPALForRead();
    const float *prev_w       = prev_weights->getPPALForRead();
    
    // Podriamos considerar hacer esto con cblas... aceleraria mucho.
    for (unsigned int i = 0; i < total_size; i++) {
      other_prev_w[i] = prev_w[i];
      other_w[i]      = w[i];
    }
    return conn;
  }

}
