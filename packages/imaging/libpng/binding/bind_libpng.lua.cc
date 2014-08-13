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

//BIND_HEADER_H
#include "libpng.h"
#include "constString.h"
#include "bind_image_RGB.h"
//BIND_END

//BIND_HEADER_C
//BIND_END


//BIND_FUNCTION libpng.read
{
  april_utils::constString cs;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  LUABIND_GET_PARAMETER(1,constString,cs);
  const char *filename = (const char *)cs;

  ImageFloatRGB *res = imaging::LibPNG::readPNG(filename);

  if (res == NULL) {
    LUABIND_ERROR("libpng.read failed");
  }

  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END

//BIND_FUNCTION libpng.write
{
  ImageFloatRGB *img;
  april_utils::constString cs;
  LUABIND_CHECK_ARGN(==, 2);
  
  LUABIND_CHECK_PARAMETER(2, string);
  LUABIND_GET_PARAMETER(1,ImageFloatRGB,img);
  LUABIND_GET_PARAMETER(2,constString,cs);
  const char *filename = (const char *)cs;

  bool ok = imaging::LibPNG::writePNG(img, filename);
  
  if (!ok) {
    LUABIND_ERROR("libpng.write failed");
  }
}
//BIND_END
