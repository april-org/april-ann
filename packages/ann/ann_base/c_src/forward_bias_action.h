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
#ifndef FORWARDBIASACTION_H
#define FORWARDBIASACTION_H

#include "action.h"
#include "actunit.h"
#include "connection.h"
#include "error_print.h"

namespace ANN {

  class ForwardBiasAction : public Action {
    ActivationUnits *outputs;
    Connections     *bias_vector;
    float learning_rate, momentum;

  public:
    ForwardBiasAction(const ANNConfiguration &conf,
		      ActivationUnits *outputs,
		      Connections *bias_vector);
    virtual ~ForwardBiasAction();
    virtual void doForward();
    virtual void doBackward();
    virtual Action *clone(hash<void*,void*> &clone_dict,
			  const ANNConfiguration &conf);
    virtual void setOption(const char *name, double value);
    virtual bool hasOption(const char *name);
    virtual double getOption(const char *name);
    void transferFanInToConnections();
  };
}

#endif // FORWARDBIASACTION_H
