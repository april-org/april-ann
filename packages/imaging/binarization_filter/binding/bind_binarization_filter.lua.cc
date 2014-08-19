/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco
 * Zamora-Martinez, Joan Pastor-Pellicer
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

//BIND_HEADER_C
//BIND_END

//BIND_HEADER_H
#include "bind_image.h"
#include "binarization.h"
#include "utilMatrixFloat.h"
using namespace imaging;
//BIND_END

//BIND_LUACLASSNAME ImageFloat Image

//BIND_METHOD ImageFloat binarize_niblack
{
  int radius;
  float k, minThreshold, maxThreshold;
  LUABIND_CHECK_ARGN(==, 4);
  LUABIND_GET_PARAMETER(1, int, radius);
  LUABIND_GET_PARAMETER(2, float, k);
  LUABIND_GET_PARAMETER(3, float, minThreshold);
  LUABIND_GET_PARAMETER(4, float, maxThreshold);
  if (radius < 1)
    LUABIND_ERROR("median filter, radius must be > 0");
  LUABIND_RETURN(ImageFloat, binarize_niblack(obj,radius, k, minThreshold, maxThreshold));
}
//BIND_END

//BIND_METHOD ImageFloat binarize_niblack_simple
{
  int radius;
  float k, minThreshold, maxThreshold;
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_GET_PARAMETER(1, int, radius);
  LUABIND_GET_OPTIONAL_PARAMETER(2,float, k, 0.2);
  if (radius < 1)
    LUABIND_ERROR("median filter, radius must be > 0");
  LUABIND_RETURN(ImageFloat, binarize_niblack_simple(obj,radius, k));
}
//BIND_END

//BIND_METHOD ImageFloat binarize_sauvola
{
  int radius;
  int r;
  float k, minThreshold, maxThreshold;
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_GET_PARAMETER(1, int, radius);
  LUABIND_GET_OPTIONAL_PARAMETER(2,float, k, 0.5);
  LUABIND_GET_OPTIONAL_PARAMETER(3,float, r, 128);
  
  if (radius < 1)
    LUABIND_ERROR("median filter, radius must be > 0");
  LUABIND_RETURN(ImageFloat, binarize_sauvola(obj,radius, k, r));
}
//BIND_END
//BIND_METHOD ImageFloat binarize_otsus
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(ImageFloat, binarize_otsus(obj));
}
//BIND_END

//BIND_METHOD ImageFloat binarize_threshold
{
  double threshold;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, double, threshold);
  LUABIND_RETURN(ImageFloat, binarize_threshold(obj, threshold));
}
//BIND_END

