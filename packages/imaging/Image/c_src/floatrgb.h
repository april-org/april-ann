/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Jorge Gorbe Moya
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
#ifndef FLOATRGB_H
#define FLOATRGB_H

#include "clamp.h"
#include "matrix_ext.h"

namespace Imaging {

  struct FloatRGB
  {
    float r,g,b;

    FloatRGB(): r(0.0f), g(0.0f), b(0.0f) {}
    FloatRGB(float r, float g, float b):r(r), g(g), b(b) {}
    explicit FloatRGB(float gray): r(gray), g(gray), b(gray) {}
    float to_grayscale() { return 0.3*r+0.59*g+0.11*b; }

  };

  FloatRGB operator + (FloatRGB x, FloatRGB y);
  FloatRGB operator - (FloatRGB x, FloatRGB y);
  FloatRGB operator * (FloatRGB x, FloatRGB y);
  FloatRGB operator / (FloatRGB x, FloatRGB y);

  FloatRGB operator + (FloatRGB x, float y);
  FloatRGB operator - (FloatRGB x, float y);
  FloatRGB operator * (FloatRGB x, float y);
  FloatRGB operator / (FloatRGB x, float y);

  FloatRGB operator + (float x, FloatRGB y);
  FloatRGB operator - (float x, FloatRGB y);
  FloatRGB operator * (float x, FloatRGB y);
  FloatRGB operator / (float x, FloatRGB y);

  FloatRGB & operator += (FloatRGB &x, FloatRGB y);
  FloatRGB & operator -= (FloatRGB &x, FloatRGB y);
  FloatRGB & operator *= (FloatRGB &x, FloatRGB y);
  FloatRGB & operator /= (FloatRGB &x, FloatRGB y);

  FloatRGB & operator += (FloatRGB &x, float y);
  FloatRGB & operator -= (FloatRGB &x, float y);
  FloatRGB & operator *= (FloatRGB &x, float y);
  FloatRGB & operator /= (FloatRGB &x, float y);

} // namespace Imaging

namespace AprilUtils{
  template<> Imaging::FloatRGB clamp<Imaging::FloatRGB>(Imaging::FloatRGB val,
                                                        Imaging::FloatRGB lower,
                                                        Imaging::FloatRGB upper);
}

namespace AprilMath {
  namespace MatrixExt {
    namespace BLAS {
      
      template<>
      Basics::Matrix<Imaging::FloatRGB> *matCopy(Basics::Matrix<Imaging::FloatRGB> *dst,
                                                 const Basics::Matrix<Imaging::FloatRGB> *src);
      
    }
    namespace Initializers {

      template<>
      Basics::Matrix<Imaging::FloatRGB> *matFill(Basics::Matrix<Imaging::FloatRGB> *obj,
                                                 const Imaging::FloatRGB value);
    }

    namespace Operations {
      
      template <>
      Basics::Matrix<Imaging::FloatRGB> *matComplement(Basics::Matrix<Imaging::FloatRGB> *src,
                                                       Basics::Matrix<Imaging::FloatRGB> *dst);
    }
  }
}

#endif
