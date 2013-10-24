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
#ifndef CONNECTION_H
#define CONNECTION_H

#include <cstring>
#include "aligned_memory.h"
#include "swap.h"
#include "referenced.h"
#include "MersenneTwister.h"
#include "matrixFloat.h"
#include "error_print.h"
#include "maxmin.h"

using april_utils::max;

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
  
  class Connections : public Referenced {
  protected:
    MatrixFloat *weights;
    MatrixFloat *prev_weights;
    // Counts the number of ANN components which shares this weight matrices
    unsigned int shared_count;
    
  public:
    static const double weightnearzero;
    
    Connections(unsigned int num_inputs, unsigned int num_outputs,
		const MatrixFloat *w=0, const MatrixFloat *oldw=0);
    ~Connections();
    
    // This method must be executed during the forward step
    void resetSharedCount() {
      shared_count = 0;
    }
    
    // This method must be executed during the backprop step
    void addToSharedCount(unsigned int count=1) {
      shared_count += count;
    }

    //
    unsigned int getSharedCount() const {
      if (shared_count == 0)
	ERROR_EXIT(128, "Found ZERO in shared_count of connections, check that "
		   "all ANN components are using properly resetSharedCount() "
		   "and addToSharedCount(...) methods\n");
      return shared_count;
    }
    
    //
    unsigned int getInputSize()  const {
      return static_cast<unsigned int>(weights->getDimSize(1));
    }
    unsigned int getNumInputs()  const {
      return static_cast<unsigned int>(weights->getDimSize(1));
    }
    unsigned int getOutputSize() const {
      return static_cast<unsigned int>(weights->getDimSize(0));
    }
    unsigned int getNumOutputs() const {
      return static_cast<unsigned int>(weights->getDimSize(0));
    }
    
    unsigned int size() const;
    void         pruneSubnormalAndCheckNormal();
    MatrixFloat *getPtr();
    MatrixFloat *getPrevPtr();
    
    // INTERFAZ A IMPLEMENTAR
    bool checkInputOutputSizes(unsigned int input_size,
			       unsigned int output_size) const;
    void randomizeWeights(MTRand *rnd, float low, float high);
    void randomizeWeightsAtColumn(unsigned int col,
				  MTRand *rnd,
				  float low, float high);
    // Carga/guarda los pesos de la matriz data comenzando por la
    // posicion first_weight_pos. Devuelve la suma del numero de pesos
    // cargados/salvados y first_weight_pos. En caso de error,
    // abortara el programa con un ERROR_EXIT
    unsigned int loadWeights(MatrixFloat *data,
			     MatrixFloat *old_data,
			     unsigned int first_weight_pos,
			     unsigned int column_size);
    
    unsigned int copyWeightsTo(MatrixFloat *data,
			       MatrixFloat *old_data,
			       unsigned int first_weight_pos,
			       unsigned int column_size);
    // para hacer copias
    Connections *clone();

    void printDebug();

    char *toLuaString();
    
    void swap();
  };
}
#endif
