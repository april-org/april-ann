/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef GAMMA_VARIATE_H
#define GAMMA_VARIATE_H

#include "MersenneTwister.h"

namespace Stats {
  /**
   * @brief Computes a Gamma variate random number.
   *
   * @params rng A random numbers generator.
   * @params a A location parameter.
   * @params b A scale parameter.
   * @params c A shape parameter.
   *
   * @note The shape can be transformed into a rate computing \f$
   * \frac{1}{shape} \f$.
   *
   * @note Modification of: http://ftp.arl.mil/random/random.pdf
   */
  
  double gammaVariate(Basics::MTRand *rng, const double a, const double b,
                      const double c);
}

#endif // GAMMA_VARIATE_H
