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

namespace imaging {

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

} // namespace imaging

namespace april_utils{
  template<> imaging::FloatRGB clamp<imaging::FloatRGB>(imaging::FloatRGB val,
                                                        imaging::FloatRGB lower,
                                                        imaging::FloatRGB upper)
  {
    float r = clamp(val.r, lower.r, upper.r);
    float g = clamp(val.g, lower.g, upper.g);
    float b = clamp(val.b, lower.b, upper.b);
    
    return imaging::FloatRGB(r,g,b);
  }
}
