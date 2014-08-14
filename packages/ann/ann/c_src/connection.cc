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
#include "buffer_list.h"
#include "connection.h"
#include "check_floats.h"
#include "smart_ptr.h"
#include "c_string.h"
#include "swap.h"
#include "utilMatrixFloat.h"
#include "wrapper.h"

using namespace april_io;
using namespace april_math;
using namespace april_utils;
using namespace basics;

#define MAX_ITERATIONS_RANDOMIZE_LOOP 1000
// generates a random weight, checking number of iterations and weightnearzero
#define rnd_weight(rnd, value, dinf, range, weightnearzero) do {	\
    unsigned int it = 0;						\
    do {								\
      (value) = (rnd)->rand((range))+(dinf);				\
      it++;								\
    } while (it < MAX_ITERATIONS_RANDOMIZE_LOOP &&			\
	     fabs((value)) < (weightnearzero));				\
    if (fabs((value)) < (weightnearzero))				\
      ERROR_PRINT("# WARNING!!! Detected weightnearzero\n");		\
  } while(false);

namespace ANN {
  const double Connections::weightnearzero = 1e-7;
  
  MatrixFloat *Connections::build(unsigned int num_inputs,
				  unsigned int num_outputs) {
    int dims[2] = { static_cast<int>(num_outputs),
		    static_cast<int>(num_inputs) };
    MatrixFloat *weights = new MatrixFloat(2, dims, CblasColMajor);
    if (weights == 0)
      ERROR_EXIT(130, "Impossible to allocate memory\n");
    return weights;
  }
  
  bool Connections::checkInputOutputSizes(const MatrixFloat *weights,
					  unsigned int input_size,
					  unsigned int output_size) {
    if (getOutputSize(weights) != output_size) {
      ERROR_PRINT("Incorrect output size!!!\n");
      return false;
    }
    if (getInputSize(weights) != input_size) {
      ERROR_PRINT("Incorrect input size!!!\n");
      return false;
    }
    return true;
  }

  // Crea de forma aleatoria el conjunto de pesos con valores en el
  // rango [low, high]
  void Connections::randomizeWeights(MatrixFloat *weights,
				     MTRand *rnd, float low, float high) {
    double dinf = low;
    double dsup = high;
    
    // assert to avoid nearzero weights
    if (fabs(dinf) < weightnearzero) dinf =  weightnearzero;
    if (fabs(dsup) < weightnearzero) dsup = -weightnearzero;
    double range  = dsup - dinf;
    MatrixFloat::iterator w_it(weights->begin());
    while(w_it != weights->end()) {
      rnd_weight(rnd, *w_it, dinf, range, weightnearzero);
      ++w_it;
    }
  }
    
  void Connections::randomizeWeightsAtColumn(MatrixFloat *weights,
					     unsigned int col,
					     MTRand *rnd,
					     float low, float high) {
    double dinf = low;
    double dsup = high;

    // assert to avoid nearzero weights
    april_assert(fabs(dinf) > weightnearzero);
    april_assert(fabs(dsup) > weightnearzero);
    double range  = dsup - dinf;
    MatrixFloat::iterator w_it(weights->iteratorAt(col,0));
    MatrixFloat::iterator end(weights->iteratorAt(col+1,0));
    while(w_it != end) {
      rnd_weight(rnd, *w_it, dinf, range, weightnearzero);
      ++w_it;
    }
  }
  
  unsigned int Connections::loadWeights(MatrixFloat *weights,
					MatrixFloat *data,
					unsigned int first_weight_pos,
					unsigned int column_size) {
    const unsigned int num_outputs = static_cast<unsigned int>(weights->getDimSize(0));
    const unsigned int num_inputs  = static_cast<unsigned int>(weights->getDimSize(1));
    const unsigned int total_size  = static_cast<unsigned int>(weights->size());
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*num_outputs +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size()))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size(), min_size);
    if (!data->isSimple())
      ERROR_EXIT(128, "Matrices need to be simple (contiguous "
		 "and in row-major)\n");
    unsigned int current_w_pos = first_weight_pos;
    MatrixFloat::iterator w_it(weights->begin());
    for (unsigned int j=0; j<num_outputs; ++j) {
      for (unsigned int i=0; i<num_inputs; ++i) {
	*w_it      = (*data)[current_w_pos+i];
	++w_it;
      }
      current_w_pos += column_size;
    }
    return current_w_pos;
  }

  unsigned int Connections::copyWeightsTo(MatrixFloat *weights,
					  MatrixFloat *data,
					  unsigned int first_weight_pos,
					  unsigned int column_size) {
    const unsigned int num_outputs = static_cast<unsigned int>(weights->getDimSize(0));
    const unsigned int num_inputs  = static_cast<unsigned int>(weights->getDimSize(1));
    const unsigned int total_size  = static_cast<unsigned int>(weights->size());
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*num_outputs +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size()))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size(), min_size);
    if (!data->isSimple())
      ERROR_EXIT(128, "Matrices need to be simple (contiguous "
		 "and in row-major)\n");    
    unsigned int current_w_pos = first_weight_pos;
    MatrixFloat::const_iterator w_it(weights->begin());
    for (unsigned int j=0; j<num_outputs; ++j) {
      for (unsigned int i=0; i<num_inputs; ++i) {
	(*data)[current_w_pos+i]     = *w_it;
	++w_it;
      }
      current_w_pos += column_size;
    }
    return current_w_pos;
  }
  
  char *Connections::toLuaString(MatrixFloat *weights) {
    UniquePtr<CStringStream> stream(new CStringStream());
    stream->put("matrix.fromString[[");
    writeMatrixToStream(weights, stream.get(), false);
    stream->put("]]\0", 3); // forces a \0 at the end of the buffer
    return stream->releaseString();
  }
}
