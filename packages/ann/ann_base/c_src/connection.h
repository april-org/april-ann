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
#include "gpu_mirrored_memory_block.h"
#include "aligned_memory.h"
#include "swap.h"
#include "constants.h"
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
    FloatGPUMirroredMemoryBlock *weights;
    FloatGPUMirroredMemoryBlock *prev_weights;
    unsigned int total_size;
    /// numero de inputs y numero de outputs
    unsigned int num_inputs, num_outputs;
    // contador de referencias
    unsigned int num_references;
    /// numero de veces que se ha llamado al metodo
    /// update_weights_call, se inicia a 0 cuando este valor llega a
    /// getNumReferences()
    unsigned int update_weights_calls;
    
  public:
    static const double weightnearzero;
    
    Connections(unsigned int num_inputs, unsigned int num_outputs);
    ~Connections();
    
    void reset() {
      num_references       = 0;
      update_weights_calls = 0;
    }
    
    // contamos el numero de veces que nos referencian, asi sabemos si
    // la conexion es compartida por mas de una accion
    void         countReference();
    unsigned int getNumReferences() const;
    unsigned int getInputSize()  const { return num_inputs; }
    unsigned int getNumInputs()  const { return num_inputs; }
    unsigned int getOutputSize() const { return num_outputs; }
    unsigned int getNumOutputs() const { return num_outputs; }
    

    void         beginUpdate();
    bool         endUpdate(); // return true when last update call
    bool         isFirstUpdateCall();
    void         computeMomentumOnPrevVector(float momentum,
					     bool  use_cuda);
    void         computeWeightDecayOnPrevVector(float c_weight_decay,
						bool  use_cuda);
    void         copyToPrevVector(bool use_cuda);
    unsigned int size() const;
    void         pruneSubnormalAndCheckNormal();
    FloatGPUMirroredMemoryBlock *getPtr();
    FloatGPUMirroredMemoryBlock *getPrevPtr();
    
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
    
    unsigned int getNumWeights() const {
      return total_size;
    }
    
    void scale(float alpha);

    void printDebug();
  };
}
#endif
