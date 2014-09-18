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

namespace ANN {

  /// The class Connections is an static class. It is a helper to build,
  /// initialize, and query matrix instances interpreted as connection weights.
  class Connections : public Referenced {
  public:
    static const double weightnearzero;
    static Basics::MatrixFloat *build(unsigned int num_inputs,
                                      unsigned int num_outputs);
    //
    static unsigned int getInputSize(const Basics::MatrixFloat *weights) {
      return static_cast<unsigned int>(weights->getDimSize(1));
    }
    static unsigned int getNumInputs(const Basics::MatrixFloat *weights) {
      return static_cast<unsigned int>(weights->getDimSize(1));
    }
    static unsigned int getOutputSize(const Basics::MatrixFloat *weights) {
      return static_cast<unsigned int>(weights->getDimSize(0));
    }
    static unsigned int getNumOutputs(const Basics::MatrixFloat *weights) {
      return static_cast<unsigned int>(weights->getDimSize(0));
    }
    
    static bool checkInputOutputSizes(const Basics::MatrixFloat *weights,
				      unsigned int input_size,
				      unsigned int output_size);

    static void randomizeWeights(Basics::MatrixFloat *weights,
				 Basics::MTRand *rnd, float low, float high);
    static void randomizeWeightsAtColumn(Basics::MatrixFloat *weights,
					 unsigned int col,
					 Basics::MTRand *rnd,
					 float low, float high);
    // Carga/guarda los pesos de la matriz data comenzando por la
    // posicion first_weight_pos. Devuelve la suma del numero de pesos
    // cargados/salvados y first_weight_pos. En caso de error,
    // abortara el programa con un ERROR_EXIT
    static unsigned int loadWeights(Basics::MatrixFloat *weights,
				    Basics::MatrixFloat *data,
				    unsigned int first_weight_pos,
				    unsigned int column_size);
    
    static unsigned int copyWeightsTo(Basics::MatrixFloat *weights,
				      Basics::MatrixFloat *data,
				      unsigned int first_weight_pos,
				      unsigned int column_size);
    
    static char *toLuaString(Basics::MatrixFloat *weights);
  };
}
#endif
