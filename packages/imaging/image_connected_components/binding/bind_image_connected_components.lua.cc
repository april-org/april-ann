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
#include "bind_matrix_int32.h"
#include "image_connected_components.h"
#include "utilImageFloat.h"
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
  LUABIND_RETURN(ImageConnectedComponents, obj);
}
//BIND_END

//BIND_METHOD ImageConnectedComponents get_size
{
 LUABIND_RETURN(int,obj->size);
}
//BIND_END


//BIND_METHOD ImageConnectedComponents get_pixel_matrix
{
  LUABIND_RETURN(MatrixInt32, obj->getPixelMatrix());
}
//BIND_END

