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
//BIND_HEADER_H
#include "affine_transform.h"
#include "utilMatrixFloat.h"
#include "bind_matrix.h"
//BIND_END


//BIND_LUACLASSNAME MatrixFloat matrix
//BIND_LUACLASSNAME AffineTransform2D AffineTransform2D
//BIND_CPP_CLASS AffineTransform2D
//BIND_SUBCLASS_OF AffineTransform2D MatrixFloat


//BIND_CONSTRUCTOR AffineTransform2D
{
  AffineTransform2D *obj;
  
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  if (argn == 0) {
    obj = new AffineTransform2D();
  }
  else {
    MatrixFloat *mat;
    LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
    if (mat->numDim != 2 || mat->matrixSize[0] != 3 || mat->matrixSize[1] != 3) {
      LUABIND_ERROR("2D affine transform matrix must be 3x3");
    }

    if (mat->data[6] != 0 || mat->data[7] != 0 || mat->data[8] != 1) {
      LUABIND_FERROR3("Bottom row of the 2D affine transform matrix must be [0 0 1] "
                      "([%f %f %f] found)", mat->data[6], mat->data[7], mat->data[8]);
    }
    obj = new AffineTransform2D(mat);
  }

  LUABIND_RETURN(AffineTransform2D, obj);
}
//BIND_END

//BIND_DESTRUCTOR AffineTransform2D
{
}
//BIND_END

//BIND_METHOD AffineTransform2D accumulate
{
  LUABIND_CHECK_ARGN(==, 1);
  AffineTransform2D *other;
  LUABIND_GET_PARAMETER(1, AffineTransform2D, other);

  obj->accumulate(other);

  LUABIND_RETURN(AffineTransform2D, obj);
}
//BIND_END

//BIND_METHOD AffineTransform2D rotate
{
  int argn = lua_gettop(L);
  if (argn == 1) {
    float angle;
    LUABIND_GET_PARAMETER(1, float, angle);
    obj->rotate(angle);
  }
  else if (argn == 3) {
    float angle, center_x, center_y;
    LUABIND_GET_PARAMETER(1, float, angle);
    LUABIND_GET_PARAMETER(2, float, center_x);
    LUABIND_GET_PARAMETER(3, float, center_y);
    obj->rotate(angle, center_x, center_y);
  }
  else {
    LUABIND_ERROR("rotate accepts only 1 or 3 arguments");
  }
  LUABIND_RETURN(AffineTransform2D, obj);
}
//BIND_END

//BIND_METHOD AffineTransform2D scale
{
  LUABIND_CHECK_ARGN(==, 2);

  float x, y;
  LUABIND_GET_PARAMETER(1, float, x);
  LUABIND_GET_PARAMETER(2, float, y);
  obj->scale(x, y);
  LUABIND_RETURN(AffineTransform2D, obj);
}
//BIND_END

//BIND_METHOD AffineTransform2D translate
{
  LUABIND_CHECK_ARGN(==, 2);

  float x, y;
  LUABIND_GET_PARAMETER(1, float, x);
  LUABIND_GET_PARAMETER(2, float, y);
  obj->translate(x, y);
  LUABIND_RETURN(AffineTransform2D, obj);
}
//BIND_END

//BIND_METHOD AffineTransform2D shear
{
  LUABIND_CHECK_ARGN(==, 2);

  float angle_x, angle_y;
  LUABIND_GET_PARAMETER(1, float, angle_x);
  LUABIND_GET_PARAMETER(2, float, angle_y);
  obj->shear(angle_x, angle_y);
  LUABIND_RETURN(AffineTransform2D, obj);
}
//BIND_END

//BIND_METHOD AffineTransform2D transform
{
  LUABIND_CHECK_ARGN(==, 2);
  float x, y, dstx, dsty;
  LUABIND_GET_PARAMETER(1, float, x);
  LUABIND_GET_PARAMETER(2, float, y);

  obj->transform(x,y, &dstx, &dsty);

  LUABIND_RETURN(float, dstx);
  LUABIND_RETURN(float, dsty);
}
//BIND_END




