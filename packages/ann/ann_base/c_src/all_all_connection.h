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
#ifndef ALLALLCONNECTIONS_H
#define ALLALLCONNECTIONS_H

#include "connection.h"

namespace ANN {
  class AllAllConnections : public Connections {
    unsigned int num_inputs, num_outputs;
  public:
    AllAllConnections(unsigned int num_inputs,
		      unsigned int num_outputs);
    ~AllAllConnections() { }

    bool checkInputOutputSizes(ActivationUnits *input,
			       ActivationUnits *output) const;
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
    Connections *clone();
  };
}

#endif // ALLALLCONNECTIONS_H
