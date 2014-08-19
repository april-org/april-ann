/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera
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
#ifndef DICE_H
#define DICE_H

#include "referenced.h"
#include "MersenneTwister.h"

namespace basics {

  class dice : public Referenced {
    int outcomes;
    double *threshold;
  public:
    dice(int outcom, double *prob);
    ~dice();
    int get_outcomes() const { return outcomes; }
    int thrown(MTRand *generator);
  };

} // namespace basics

#endif // DICE_H
