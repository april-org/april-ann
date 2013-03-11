/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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

//BIND_HEADER_H
#include <errno.h>
#include <stdio.h>
#include "image_cleaning.h"
#include "bind_dataset.h"
#include "bind_image.h"
#include "bind_matrix.h"
//BIND_END


/////////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ImageHistogram image.image_histogram
//BIND_CPP_CLASS    ImageHistogram

//BIND_CONSTRUCTOR ImageHistogram
//DOC_BEGIN
// ImageHistogram()
// @param number of gray_levels
// @param number to compute the radius
//DOC_END
{

  LUABIND_CHECK_ARGN(==, 2);

  int gray_levels;
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_PARAMETER(2, int, gray_levels);

  ImageHistogram *obj = new ImageHistogram(img, gray_levels);
  LUABIND_RETURN(ImageHistogram, obj);
}
//BIND_END

//BIND_CONSTRUCTOR ImageHistogram
//DOC_BEGIN
// ImageHistogram()
// @param number of gray_levels
// @param number to compute the radius
//DOC_END
{

  LUABIND_CHECK_ARGN(==, 2);

  int gray_levels;
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_PARAMETER(2, int, gray_levels);

  ImageHistogram *obj = new ImageHistogram(img, gray_levels);
  LUABIND_RETURN(ImageHistogram, obj);
}
//BIND_END

//BIND_METHOD ImageHistogram generate_window_histogram
//DOC_BEGIN
// Returns the number of gray levels in the class
//
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  int radius;
  LUABIND_GET_PARAMETER(1, int, radius);
  LUABIND_RETURN(MatrixFloat, obj->generateWindowHistogram(radius));
}
//BIND_END

//BIND_METHOD ImageHistogram get_integral_histogram
//DOC_BEGIN
// Returns the 3-D integral matrix with the histogram
//
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(MatrixFloat, obj->getIntegralHistogram());
}
//BIND_END
//////////////////////////////////////////////////////////////////////

