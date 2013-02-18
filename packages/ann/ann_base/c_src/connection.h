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
#ifndef CONNECTION_H
#define CONNECTION_H

#include <cstring>
#include "aligned_memory.h"
#include "swap.h"

#include "constants.h"
#include "actunit.h"
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

    unsigned int fanin;
    
  public:
    static const double weightnearzero;
    
    Connections(unsigned int total_size,
		unsigned int num_inputs, unsigned int num_outputs);
    virtual ~Connections();
    
    // contamos el numero de veces que nos referencian, asi sabemos si
    // la conexion es compartida por mas de una accion
    void         countReference();
    unsigned int getNumReferences() const;
    virtual void         beginUpdate();
    virtual void         endUpdate(); // return true when last update call
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
    virtual bool checkInputOutputSizes(ActivationUnits *input,
				       ActivationUnits *output) const = 0;
    virtual void randomizeWeights(MTRand *rnd, float low, float high,
				  bool use_fanin) = 0;
    virtual void randomizeWeightsAtColumn(unsigned int col,
					  MTRand *rnd,
					  float low, float high,
					  bool use_fanin) = 0;
    // Carga/guarda los pesos de la matriz data comenzando por la
    // posicion first_weight_pos. Devuelve la suma del numero de pesos
    // cargados/salvados y first_weight_pos. En caso de error,
    // abortara el programa con un ERROR_EXIT
    virtual unsigned int loadWeights(MatrixFloat *data,
				     MatrixFloat *old_data,
				     unsigned int first_weight_pos,
				     unsigned int column_size) = 0;

    virtual unsigned int copyWeightsTo(MatrixFloat *data,
				       MatrixFloat *old_data,
				       unsigned int first_weight_pos,
				       unsigned int column_size) = 0;
    // para hacer copias
    virtual Connections *clone() = 0;
    
    virtual unsigned int getNumWeights() const {
      return total_size;
    }
    
    virtual unsigned int getNumInputs() const {
      return num_inputs;
    }
    virtual unsigned int getNumOutputs() const {
      return num_outputs;
    }
    
    void setFanIn(unsigned int value) { fanin = max(fanin, value); }
    
  };
}
#endif
