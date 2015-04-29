/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#ifndef PRELUACTFCOMPONENT_H
#define PRELUACTFCOMPONENT_H

#include "activation_function_component.h"
#include "ann_component.h"
#include "gpu_mirrored_memory_block.h"
#include "smart_ptr.h"

namespace ANN {

  /**
   * @brief Component for the PReLU activation function.
   *
   * @see http://arxiv.org/pdf/1502.01852.pdf
   */
  class PReLUActfANNComponent : public ActivationFunctionANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(PReLUActfANNComponent);
    AprilUtils::SharedPtr<Basics::MatrixFloat> weights;
    unsigned int size;
    bool shared;
  protected:
    virtual void applyActivation(Basics::MatrixFloat *input_units,
                                 Basics::MatrixFloat *output_units);
    virtual void multiplyDerivatives(Basics::MatrixFloat *input_units,
				     Basics::MatrixFloat *output_units,
				     Basics::MatrixFloat *input_errors,
				     Basics::MatrixFloat *output_errors);
  public:
    PReLUActfANNComponent(bool shared = false, unsigned int size = 0,
                          const char *name = 0, const char *weights_name = 0);
    virtual ~PReLUActfANNComponent();
    virtual ANNComponent *clone();

    virtual char *toLuaString();

    virtual void computeGradients(const char *weights_name,
                                  AprilUtils::LuaTable &weight_grads);

    virtual void build(unsigned int _input_size, unsigned int _output_size,
                       AprilUtils::LuaTable &weights_dict,
                       AprilUtils::LuaTable &components_dict);

    virtual void reset(unsigned int it=0);
  };
}

#endif // PRELUACTFCOMPONENT_H
