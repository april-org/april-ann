/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#ifndef CONVOLUTIONBIASANNCOMPONENT_H
#define CONVOLUTIONBIASANNCOMPONENT_H  

#include "token_matrix.h"
#include "cblas_headers.h"
#include "ann_component.h"
#include "connection.h"

namespace ANN {

  /// A component which adds a convolutional bias given the number of output
  /// planes. Exists one bias per each output plane.
  class ConvolutionBiasANNComponent : public ANNComponent {
    TokenMatrixFloat *input, *error, *output;
    Connections *bias_vector;
    MatrixFloat *bias_matrix; // rewrapping of bias_vector matrix to fits at
			      // input window sizes

    /// The number of convolutions computed during last forward
    int number_input_windows;
    
    // parameters of the convolution bias
    unsigned int hidden_size; // number of planes
    int num_dims;
    int *window_size;
    int *window_step;
    int *window_num_steps;
    
    void initializeArrays(const int *input_dims);
    MatrixFloat *prepareBiasBunch();

  protected:

    virtual void computeGradients(MatrixFloat*& grads_mat);

  public:
    ConvolutionBiasANNComponent(int input_num_dims,
				unsigned int num_output_planes,   // hidden layer size
				const char *name=0,
				const char *bias_name=0);
    virtual ~ConvolutionBiasANNComponent();
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error; }
    virtual Token *getErrorOutput() { return error; }
    virtual Token *doForward(Token* input, bool during_training);
    virtual Token *doBackprop(Token *input_error);
    virtual void   reset();
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    virtual void copyWeights(hash<string,Connections*> &weights_dict);
    virtual char *toLuaString();

  };
}

#endif // CONVOLUTIONANNCOMPONENT_H
