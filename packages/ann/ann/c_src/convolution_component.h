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
#ifndef CONVOLUTIONANNCOMPONENT_H
#define CONVOLUTIONANNCOMPONENT_H  

#include "token_matrix.h"
#include "cblas_headers.h"
#include "ann_component.h"
#include "connection.h"

namespace ANN {

  /// A component which computes a convolutional layer using given kernel size
  /// and step, and the given number of output planes.
  class ConvolutionANNComponent : public ANNComponent {
    TokenMatrixFloat *input, *error_input, *output, *error_output;
    MatrixFloat *weights_matrix;
    unsigned int num_updates_from_last_prune;
    
    // parameters of the convolution
    
    /// Dimension where input planes are located
    const int input_planes_dim;
    /// The number of convolutions computed during last forward
    int number_input_windows;
    /// The size of one kernel (number of inputs of one hidden neuron)
    unsigned int kernel_size;
    /// The number of neurons (number of output planes)
    const unsigned int hidden_size;
    /// The number of expected input dimensions without count BUNCH dim (first)
    const int input_num_dims;
    /// Size at each dim of the kernel, input_num_dims + 1
    int *kernel_dims;
    /// Step at each dim of the input, input_num_dims + 1
    int *kernel_step;
    /// Size at each dim of the output, input_num_dims + 1
    int *output_dims; // first is BUNCH, second is number of output planes)
    // INPUT SLIDING WINDOW SECTION
    /// Size of the convolution window, input_num_dims + 1
    int *input_window_size;
    /// Number of steps of the convolution window, input_num_dims + 1
    int *input_window_num_steps;
    /// Order for traversing the input data, prepared for col-major order
    int *input_window_order_step;
    /// Translates the input window into a bi-dimensional matrix
    int *input_window_rewrap;
    // OUTPUT SLIDING WINDOW SECTION
    /// Size of the convolution window, input_num_dims + 1
    int *output_window_size;
    /// Step between convolution windows, input_num_dims + 1
    int *output_window_step;
    /// Number of steps of the convolution window, input_num_dims + 1
    int *output_window_num_steps;
    /// Order for traversing the input data, prepared for col-major order
    int *output_window_order_step;
    /// Translates the output window into a bi-dimensional matrix
    int *output_window_rewrap;
    
    MatrixFloat *getRewrappedMatrix(MatrixFloat *w,
				    const int *rewrap_size,
				    const int N,
				    bool copy) const {
      MatrixFloat *w_flattened;
      if (w->getIsContiguous()) w_flattened = w->rewrap(rewrap_size, N);
      else {
	MatrixFloat *w_clone;
	if (copy) w_clone = w->clone();
	else w_clone = w->cloneOnlyDims();
	IncRef(w_clone);
	w_flattened = w_clone->rewrap(rewrap_size, N);
	DecRef(w_clone);
      }
      return w_flattened;
    }
    
    void initializeArrays(const int *input_dims);
    
  protected:

    virtual void computeGradients(MatrixFloat*& grads_mat);

  public:
    ConvolutionANNComponent(int input_num_dims,
			    const int *_kernel_dims,  // input_num_dims
			    const int *_kernel_step,  // step
			    const int input_planes_dim, // dimension where input
						        // planes are located
			    int num_output_planes,      // hidden layer size
			    const char *name=0, const char *weights_name=0);
    virtual ~ConvolutionANNComponent();
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }
    virtual void precomputeOutputSize(const vector<unsigned int> &input_size,
				      vector<unsigned int> &output_size) {
      output_size.clear();
      output_size.push_back(hidden_size);
      for (int i=1; i<input_planes_dim; ++i)
	output_size.push_back((input_size[i-1]-kernel_dims[i])/kernel_step[i]+1);
      for (int i=input_planes_dim+1; i<=input_num_dims; ++i)
	output_size.push_back((input_size[i-1]-kernel_dims[i])/kernel_step[i]+1);
    }
    virtual Token *doForward(Token* input, bool during_training);
    virtual Token *doBackprop(Token *input_error);
    virtual void   reset(unsigned int it=0);
    virtual ANNComponent *clone();
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,MatrixFloat*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    virtual void copyWeights(hash<string,MatrixFloat*> &weights_dict);

    virtual char *toLuaString();

  };
}

#endif // CONVOLUTIONANNCOMPONENT_H
