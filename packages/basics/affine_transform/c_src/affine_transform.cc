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
#include "affine_transform.h"
#include <cmath>

const int AffineTransform2D::dimensions[2] = {3,3};

/// Identity transform
AffineTransform2D::AffineTransform2D():
  Matrix<float>(2, dimensions, 0.0f)
{
  data[0] = 1.0f;
  data[4] = 1.0f;
  data[8] = 1.0f;
}

AffineTransform2D::AffineTransform2D(Matrix<float> *mat): Matrix<float>(mat)
{
  assert("AffineTransform2D: Matrix must be 3x3" && 
         numDim == 2 && 
         matrixSize[0] == 3 &&
         matrixSize[1] == 3);

  assert("AffineTransform2D: Last row must be [0 0 1]" &&
         data[6] == 0 && data[7] == 0 && data[8] == 1);

}

AffineTransform2D *AffineTransform2D::accumulate(AffineTransform2D *other)
{
  Matrix<float> *temp = other->multiply(this);
  for (int i=0; i<6; i++)
    this->data[i] = temp->data[i];
  return this;
}

AffineTransform2D *AffineTransform2D::rotate(float angle)
{
  AffineTransform2D trans; // 3x3 identity
  trans.data[0] = cosf(angle);
  trans.data[1] = -sinf(angle);
  trans.data[3] = sinf(angle);
  trans.data[4] = cosf(angle);

  accumulate(&trans);
  return this;
}

AffineTransform2D *AffineTransform2D::rotate(float angle, float center_x, float center_y)
{
  translate(-center_x, -center_y);
  rotate(angle);
  translate(center_x, center_y);
  return this;
}

AffineTransform2D *AffineTransform2D::translate(float x, float y)
{
  AffineTransform2D trans; // 3x3 identity
  trans.data[2] = x;
  trans.data[5] = y;
  accumulate(&trans);
  return this;
}

AffineTransform2D *AffineTransform2D::scale(float x, float y)
{
  AffineTransform2D trans; // 3x3 identity
  trans.data[0] = x;
  trans.data[4] = y;
  accumulate(&trans);
  return this;
}

AffineTransform2D *AffineTransform2D::shear(float x, float y)
{
  AffineTransform2D trans; // 3x3 identity
  trans.data[1] = tanf(x);
  trans.data[3] = tanf(y);
  accumulate(&trans);
  return this;
}

void AffineTransform2D::transform(float x, float y, float *dst_x, float *dst_y) const
{
  float res_x = data[0]*x+data[1]*y+data[2];
  float res_y = data[3]*x+data[4]*y+data[5];

  *dst_x=res_x;
  *dst_y=res_y;
}
