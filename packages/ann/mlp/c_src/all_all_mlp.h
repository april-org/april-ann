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
#ifndef ALL_ALL_MLP_H
#define ALL_ALL_MLP_H

#include "mlp.h"

namespace ANN {

  /// Class that represents a neural network with all-all connections
  class AllAllMLP : public MLP {
    char *description;
    
  protected:
    
    void generateActionsAllAll(const char *description);
    unsigned int getNumberOfWeights();
    
  public:

    AllAllMLP(ANNConfiguration configuration);
    ~AllAllMLP();
    
    void generateAllAll(const char *str, MTRand *rnd, float low, float high);
    void generateAllAll(const char *str,
			MatrixFloat *weights_mat,
			MatrixFloat *old_weights_mat);
    
    void copyWeightsToMatrix(MatrixFloat **weights_mat,
			     MatrixFloat **old_weights_mat);
    const char *getDescription() { return description; }
    AllAllMLP *clone();
    void loadModel(const char *filename);
    void saveModel(const char *filename);
    
    void randomizeWeights(MTRand *rnd, float low, float high);
  };

}

#endif // ALL_ALL_MLP_H
