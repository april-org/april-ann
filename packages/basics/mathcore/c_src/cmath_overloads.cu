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
#include <climits>

#include "cmath_overloads.h"

namespace AprilMath {
  
  template<> char Limits<char>::min() { return CHAR_MIN; }
  template<> char Limits<char>::max() { return CHAR_MAX; }

  template<> int32_t Limits<int32_t>::min() { return INT_MIN; }
  template<> int32_t Limits<int32_t>::max() { return INT_MAX; }
  
  template<> float Limits<float>::min() { return FLT_MIN; }
  template<> float Limits<float>::max() { return FLT_MAX; }
  template<> float Limits<float>::epsilon() { return FLT_EPSILON; }
  
  template<> double Limits<double>::min() { return DBL_MIN; }
  template<> double Limits<double>::max() { return DBL_MAX; }
  template<> double Limits<double>::epsilon() { return DBL_EPSILON; }
  
  template<> ComplexF Limits<ComplexF>::min() {
    return ComplexF(FLT_MIN,FLT_MIN);
  }
  template<> ComplexF Limits<ComplexF>::max() {
    return ComplexF(FLT_MAX,FLT_MAX);
  }
  template<> ComplexF Limits<ComplexF>::epsilon() {
    return ComplexF(FLT_EPSILON,FLT_EPSILON);
  }
  
} // namespace AprilMath
