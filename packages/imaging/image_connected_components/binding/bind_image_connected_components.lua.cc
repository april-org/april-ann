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

//BIND_HEADER_H
#include <cerrno>
#include <cstdio>
#include "bind_dataset.h"
#include "bind_image.h"
#include "bind_image_RGB.h"
#include "bind_matrix_int32.h"
#include "image_connected_components.h"
#include "utilImageFloat.h"
#include "utilMatrixFloat.h"
//BIND_END

//BIND_FUNCTION image.test_connected_components
{
 const ImageFloat *img; 
 LUABIND_GET_PARAMETER(1, ImageFloat, img);
 
 ImageConnectedComponents components = ImageConnectedComponents(img);
 LUABIND_RETURN(int, components.size);
}

//BIND_END

/////////////////////////////////////////////////////////////////////////////////
//BIND_LUACLASSNAME ImageConnectedComponents image.connected_components
//BIND_CPP_CLASS    ImageConnectedComponents

//BIND_CONSTRUCTOR ImageConnectedComponents
//DOC_BEGIN
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  ImageConnectedComponents *obj = new ImageConnectedComponents(img);
  obj->getColoredImage();
  LUABIND_RETURN(ImageConnectedComponents, obj);
}
//BIND_END

//BIND_METHOD ImageConnectedComponents get_size
{
 LUABIND_RETURN(int,obj->size);
}
//BIND_END

//BIND_METHOD ImageConnectedComponents get_colored_image
{
  LUABIND_RETURN(ImageFloatRGB, obj->getColoredImage());
}
//BIND_END

//BIND_METHOD ImageConnectedComponents get_pixel_matrix
{
  LUABIND_RETURN(MatrixInt32, obj->getPixelMatrix());
}
//BIND_END

//BIND_METHOD ImageConnectedComponents connected
{
  LUABIND_CHECK_ARGN(==, 4);
  int x1, y1, x2, y2;

  LUABIND_GET_PARAMETER(1, int, x1);
  LUABIND_GET_PARAMETER(2, int, y1);
  LUABIND_GET_PARAMETER(3, int, x2);
  LUABIND_GET_PARAMETER(4, int, y2);

  LUABIND_RETURN(bool, obj->connected(x1, y1, x2, y2));
}
//BIND_END

//BIND_METHOD ImageConnectedComponents get_bounding_boxes
{
  LUABIND_CHECK_ARGN(==, 0);
  int size;  

  vector<bounding_box> *bbs = obj->getBoundingBoxes();
  size = (int) bbs->size();

  lua_createtable(L, size, 0);
  for (size_t i = 0; i < bbs->size(); ++i) {
      lua_createtable(L, 4, 0);
      lua_pushint(L,(*bbs)[i].x1);
      lua_rawseti(L, -2, 1);
      lua_pushint(L, (*bbs)[i].y1);
      lua_rawseti(L, -2, 2);
      lua_pushint(L, (*bbs)[i].x2);
      lua_rawseti(L, -2, 3);
      lua_pushint(L, (*bbs)[i].y2);
      lua_rawseti(L, -2, 4);
      lua_rawseti(L, -2, i+1);
  }
  
  delete bbs;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

