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
#include <cstdio>
#include "complex_number.h"

namespace AprilMath {
  template struct Complex<float>;
  template struct Complex<double>;
}

namespace AprilUtils {
  void aprilPrint(const AprilMath::ComplexF &v) {
    printf("%f%c%fi", v.real(), (v.img()>+0.0f)?'+':' ', v.img());
  }
  void aprilPrint(const AprilMath::ComplexD &v) {
    printf("%f%c%fi", v.real(), (v.img()>+0.0)?'+':' ', v.img());
  }

  template<>
  unsigned int binarizer::binary_size<AprilMath::ComplexF>(){
    return 10;
  }
  template<>
  unsigned int binarizer::binary_size<AprilMath::ComplexD>() {
    return 20;
  }

  template<>
  void binarizer::
  code<AprilMath::ComplexF>(const AprilMath::ComplexF &value, char *b) {
    binarizer::code_float(value.real(), b);
    binarizer::code_float(value.img(), b+5);
  }

  template<>
  void binarizer::
  code<AprilMath::ComplexD>(const AprilMath::ComplexD &value, char *b) {
    binarizer::code_double(value.real(), b);
    binarizer::code_double(value.img(), b+10);
  }

  template<>
  AprilMath::ComplexF binarizer::decode<AprilMath::ComplexF>(const char *b) {
    float real, img;
    real = binarizer::decode_float(b);
    img = binarizer::decode_float(b+5);
    return AprilMath::ComplexF(real,img);
  }

  template<>
  AprilMath::ComplexD binarizer::decode<AprilMath::ComplexD>(const char *b) {
    double real, img;
    real = binarizer::decode_float(b);
    img = binarizer::decode_float(b+5);
    return AprilMath::ComplexD(real,img);
  }
}
