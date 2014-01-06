/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
  MatrixFloat(2, dimensions)
{
  zeros();
  MatrixFloat::random_access_iterator data(this);
  data(0,0) = 1.0f;
  data(1,1) = 1.0f;
  data(2,2) = 1.0f;
}

AffineTransform2D::AffineTransform2D(MatrixFloat *mat): MatrixFloat(mat)
{
  april_assert("AffineTransform2D: Matrix must be 3x3" && 
	       numDim == 2 && 
	       matrixSize[0] == 3 &&
	       matrixSize[1] == 3);

  april_assert("AffineTransform2D: Last row must be [0 0 1]" &&
	       (*this)(2,0) == 0 && (*this)(2,1) == 0 && (*this)(2,2) == 1);

}

AffineTransform2D *AffineTransform2D::accumulate(AffineTransform2D *other)
{
  MatrixFloat *this_clone = this->clone();
  IncRef(this_clone);
  this->gemm(CblasNoTrans, CblasNoTrans,
	     1.0f, other, this_clone,
	     0.0f);
  DecRef(this_clone);
  return this;
}

AffineTransform2D *AffineTransform2D::rotate(float angle)
{
  AffineTransform2D trans; // 3x3 identity
  trans(0,0) = cosf(angle);
  trans(0,1) = -sinf(angle);
  trans(1,0) = sinf(angle);
  trans(1,1) = cosf(angle);
  
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
  trans(0,2) = x;
  trans(1,2) = y;
  accumulate(&trans);
  return this;
}

AffineTransform2D *AffineTransform2D::scale(float x, float y)
{
  AffineTransform2D trans; // 3x3 identity
  trans(0,0) = x;
  trans(1,1) = y;
  accumulate(&trans);
  return this;
}

AffineTransform2D *AffineTransform2D::shear(float x, float y)
{
  AffineTransform2D trans; // 3x3 identity
  trans(0,1) = tanf(x);
  trans(1,0) = tanf(y);
  accumulate(&trans);
  return this;
}

void AffineTransform2D::transform(float x, float y, float *dst_x, float *dst_y) const
{
  MatrixFloat::const_random_access_iterator data(this);
  float res_x = data(0,0)*x+data(0,1)*y+data(0,2);
  float res_y = data(1,0)*x+data(1,1)*y+data(1,2);
  
  *dst_x=res_x;
  *dst_y=res_y;
}
