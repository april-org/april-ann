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
#ifndef UTIL_RPROP_H
#define UTIL_RPROP_H

#include "matrixFloat.h"
namespace ANN {
  namespace optimizer {
    class UtilRProp : public Referenced {
    public:
      static void step(Basics::MatrixFloat *steps,
		       Basics::MatrixFloat *old_sign,
		       Basics::MatrixFloat *sign,
		       float eta_minus,
		       float eta_plus);
    };
  }
}

#endif // UTIL_RPROP_H
