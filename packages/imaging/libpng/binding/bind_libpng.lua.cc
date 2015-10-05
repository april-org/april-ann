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
#include "bind_april_io.h"
//BIND_END


//BIND_FUNCTION libpng.read
{
  AprilUtils::constString cs;
  LUABIND_CHECK_ARGN(==, 1);
  
  AprilUtils::SharedPtr<AprilIO::StreamInterface> stream;
  if (lua_isstring(L,1)) {
    LUABIND_GET_PARAMETER(1,constString,cs);
    const char *filename = (const char *)cs;
    stream = new AprilIO::FileStream(filename, "r");
  }
  else {
    stream = lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,1);
  }
  
  ImageFloatRGB *res = Imaging::LibPNG::readPNG(stream.get());

  if (res == NULL) {
    LUABIND_ERROR("libpng.read failed");
  }

  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END

//BIND_FUNCTION libpng.write
{
  ImageFloatRGB *img;
  AprilUtils::constString cs;
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  LUABIND_GET_PARAMETER(1,ImageFloatRGB,img);
  
  AprilUtils::SharedPtr<AprilIO::StreamInterface> stream;
  if (lua_gettop(L) == 2) {
    if (lua_isstring(L,2)) {
      LUABIND_GET_PARAMETER(2,constString,cs);
      const char *filename = (const char *)cs;
      stream = new AprilIO::FileStream(filename, "w");
    }
    else {
      stream = lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,2);
    }
  }
  else {
    stream = new AprilIO::OutputLuaStringStream(L);
  }

  bool ok = Imaging::LibPNG::writePNG(img, stream.get());

  if (!ok) {
    LUABIND_ERROR("libpng.write failed");
  }

  if (lua_gettop(L) == 1) {
    AprilIO::OutputLuaStringStream *lua_stream;
    lua_stream = (AprilIO::OutputLuaStringStream*)(stream.get());
    LUABIND_INCREASE_NUM_RETURNS( lua_stream->push(L) );
  }
  else {
    LUABIND_RETURN(StreamInterface, stream.get());
  }
}
//BIND_END
