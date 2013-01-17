/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Jorge Gorbe Moya, Salvador España-Boquera
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
#ifndef AFFINE_TRANSFORM_H
#define AFFINE_TRANSFORM_H

#include "matrix.h"

class AffineTransform2D: public Matrix<float>
{
  public:
  AffineTransform2D();
  AffineTransform2D(Matrix<float> *mat);
  ~AffineTransform2D() {}
  
  AffineTransform2D *accumulate(AffineTransform2D *other);
  AffineTransform2D *rotate(float angle);
  AffineTransform2D *rotate(float angle, float center_x, float center_y);
  AffineTransform2D *translate(float x, float y);
  AffineTransform2D *scale(float x, float y);
  AffineTransform2D *shear(float angle_x, float angle_y);

  void transform(float x, float y, float *dstx, float *dsty) const;
  private:
  static const int dimensions[2];
};

#endif

