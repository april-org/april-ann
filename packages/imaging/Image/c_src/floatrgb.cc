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
#include "floatrgb.h"
#include "matrix_operations.h"

namespace Imaging {

  FloatRGB operator + (FloatRGB x, FloatRGB y)
  {
    return FloatRGB(x.r + y.r, x.g + y.g, x.b + y.b);
  }

  FloatRGB operator - (FloatRGB x, FloatRGB y)
  {
    return FloatRGB(x.r - y.r, x.g - y.g, x.b - y.b);
  }

  FloatRGB operator * (FloatRGB x, FloatRGB y)
  {
    return FloatRGB(x.r * y.r, x.g * y.g, x.b * y.b);
  }

  FloatRGB operator / (FloatRGB x, FloatRGB y)
  {
    return FloatRGB(x.r / y.r, x.g / y.g, x.b / y.b);
  }

  FloatRGB operator + (FloatRGB x, float y)
  {
    return x+FloatRGB(y);
  }

  FloatRGB operator - (FloatRGB x, float y)
  {
    return x-FloatRGB(y);
  }

  FloatRGB operator * (FloatRGB x, float y)
  {
    return x*FloatRGB(y);
  }

  FloatRGB operator / (FloatRGB x, float y)
  {
    return x/FloatRGB(y);
  }

  FloatRGB operator + (float x, FloatRGB y)
  {
    return FloatRGB(x)+y;
  }

  FloatRGB operator - (float x, FloatRGB y)
  {
    return FloatRGB(x)-y;
  }

  FloatRGB operator * (float x, FloatRGB y)
  {
    return FloatRGB(x)*y;
  }

  FloatRGB operator / (float x, FloatRGB y)
  {
    return FloatRGB(x)/y;
  }

  FloatRGB operator += (FloatRGB x, FloatRGB y)
  {
    return (x = x + y);
  }

  FloatRGB operator -= (FloatRGB x, FloatRGB y)
  {
    return (x = x - y);
  }

  FloatRGB operator *= (FloatRGB x, FloatRGB y)
  {
    return (x = x * y);
  }

  FloatRGB operator /= (FloatRGB x, FloatRGB y)
  {
    return (x = x / y);
  }

  FloatRGB operator += (FloatRGB x, float y)
  {
    return (x = x + y);
  }

  FloatRGB operator -= (FloatRGB x, float y)
  {
    return (x = x - y);
  }

  FloatRGB operator *= (FloatRGB x, float y)
  {
    return (x = x * y);
  }

  FloatRGB operator /= (FloatRGB x, float y)
  {
    return (x = x / y);
  }

} // namespace Imaging

namespace AprilUtils{
  template<> Imaging::FloatRGB clamp<Imaging::FloatRGB>(Imaging::FloatRGB val,
                                                        Imaging::FloatRGB lower,
                                                        Imaging::FloatRGB upper)
  {
    float r = clamp(val.r, lower.r, upper.r);
    float g = clamp(val.g, lower.g, upper.g);
    float b = clamp(val.b, lower.b, upper.b);
    
    return Imaging::FloatRGB(r,g,b);
  }
}

namespace AprilMath {
  namespace MatrixExt {
    namespace Operations {
      
      template <>
      Basics::Matrix<Imaging::FloatRGB> *matCopy(Basics::Matrix<Imaging::FloatRGB> *obj,
                                                 const Basics::Matrix<Imaging::FloatRGB> *other) {
        if (!obj->sameDim(other)) ERROR_EXIT(128,"Incompatible matrix sizes\n");
        Basics::Matrix<Imaging::FloatRGB>::iterator dst_it(obj->begin());
        for (Basics::Matrix<Imaging::FloatRGB>::const_iterator src_it(other->begin());
             src_it != other->end(); ++src_it, ++dst_it) {
          *dst_it = *src_it;
        }
        return obj;
      }
      
      template <>
      Basics::Matrix<Imaging::FloatRGB> *matComplement(Basics::Matrix<Imaging::FloatRGB> *src,
                                                       Basics::Matrix<Imaging::FloatRGB> *dst) {
        if (dst == 0) dst = src;
        if (!src->sameDim(dst)) ERROR_EXIT(128,"Incompatible matrix sizes\n");
        Basics::Matrix<Imaging::FloatRGB>::iterator dst_it(dst->begin());
        for (Basics::Matrix<Imaging::FloatRGB>::const_iterator src_it(src->begin());
             src_it != dst->end(); ++src_it, ++dst_it) {
          *dst_it = 1.0f - *src_it;
        }
        return dst;
      }

      template<>
      Basics::Matrix<Imaging::FloatRGB> *matFill(Basics::Matrix<Imaging::FloatRGB> *obj,
                                                 const Imaging::FloatRGB value) {
        for (Basics::Matrix<Imaging::FloatRGB>::iterator it(obj->begin());
             it != obj->end(); ++it) {
          *it = value;
        }
        return obj;
      }
    }
  }
}
