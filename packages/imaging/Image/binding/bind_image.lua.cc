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
//BIND_HEADER_C
#include "bind_image_RGB.h"

using AprilUtils::constString;
//BIND_END

//BIND_HEADER_H
#include "utilMatrixFloat.h"
#include "utilImageFloat.h"
#include "bind_matrix.h"
#include "bind_affine_transform.h"
#include <cmath>
#include "datasetFloat.h"
#include "dataset.h"
#include "bind_dataset.h"

using namespace Imaging;
//BIND_END

//BIND_LUACLASSNAME ImageFloat Image
//BIND_CPP_CLASS ImageFloat

//BIND_CONSTRUCTOR ImageFloat
// este constructor recibe una matrix
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  
  Basics::MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  
  if (mat->getNumDim() != 2) {
    lua_pushstring(L,"image : matrix of dimension 2 is required");
    lua_error(L);    
  }

  int width=mat->getDimSize(1);
  int height=mat->getDimSize(0);
  int offsetx=0;
  int offsety=0;

  if (argn == 2 && lua_isstring(L,2)) {
    // format <width>x<height>{+-}<x>{+-}<y>
    constString cs = constString(lua_tostring(L,2),lua_strlen(L,2));
    if (!(cs.extract_int(&width, 10,  "x")  && 
	  (cs.skip(1),cs.extract_int(&height, 10, "+-")) && 
	  (cs.skip(1),cs.extract_int(&offsetx, 10, "+-")) && 
	  (cs.skip(1),cs.extract_int(&offsety)))) {
      lua_pushstring(L,"image crop: bad format string");
      lua_error(L);
    }
  }
  if (argn == 5) {
    LUABIND_GET_PARAMETER(2, int, width);
    LUABIND_GET_PARAMETER(3, int, height);
    LUABIND_GET_PARAMETER(4, int, offsetx);
    LUABIND_GET_PARAMETER(5, int, offsety);
  }

  obj = new ImageFloat(mat,width,height,offsetx,offsety);

  LUABIND_RETURN(ImageFloat, obj);
}
//BIND_END

//BIND_DESTRUCTOR ImageFloat
{
}
//BIND_END

//BIND_METHOD ImageFloat crop
{
  int argn = lua_gettop(L); // number of arguments
  int width=0,height=0,offsetx=0,offsety=0;
  if (argn == 1) {
    if (!lua_isstring(L,1)) {
      lua_pushstring(L,"image crop: 1 argument must be <width>x<height>{+-}<x>{+-}<y>");
      lua_error(L);
    }
    constString cs = constString(lua_tostring(L,1),lua_strlen(L,1));
    if (!(cs.extract_int(&width, 10, "x")  && 
	  (cs.skip(1),cs.extract_int(&height, 10, "+-")) && 
	  (cs.skip(1),cs.extract_int(&offsetx, 10, "+-")) && 
	  (cs.skip(1),cs.extract_int(&offsety)))) {
      lua_pushstring(L,"image crop: bad format string");
      lua_error(L);
    }
  } 
  else if (argn == 4) {
    LUABIND_GET_PARAMETER(1, int, width);
    LUABIND_GET_PARAMETER(2, int, height);
    LUABIND_GET_PARAMETER(3, int, offsetx);
    LUABIND_GET_PARAMETER(4, int, offsety);
  } else {
    LUABIND_ERROR("invalid number of arguments. You must use crop(\"<width>x<height>{+-}<x>{+-}<y>\") or "
        "crop(width, height, x, y).");
  }
  ImageFloat *cropimage = obj->crop(width,height,offsetx,offsety);
  
  LUABIND_RETURN(ImageFloat, cropimage);
}
//BIND_END

//BIND_METHOD ImageFloat info
{
  LUABIND_ERROR("ERROR: ImageFloat.info() is deprecated\n"
		"       Use matrix() or geometry() as needed.\n");
  LUABIND_CHECK_ARGN(==,0);
  
  LUABIND_RETURN(MatrixFloat, obj->getMatrix());
  LUABIND_RETURN(int, obj->width());
  LUABIND_RETURN(int, obj->height());
  LUABIND_RETURN(int, obj->offset_width());
  LUABIND_RETURN(int, obj->offset_height());
}
//BIND_END

//BIND_METHOD ImageFloat matrix
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(MatrixFloat, obj->getMatrix());
}
//BIND_END

//BIND_METHOD ImageFloat geometry
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(int, obj->width());
  LUABIND_RETURN(int, obj->height());
  LUABIND_RETURN(int, obj->offset_width());
  LUABIND_RETURN(int, obj->offset_height());
}
//BIND_END


//BIND_METHOD ImageFloat getpixel
{
  int x,y;

  LUABIND_CHECK_ARGN(==,2);
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  
  float pixelvalue = obj->operator()(x,y);
  
  LUABIND_RETURN(float, pixelvalue);
}
//BIND_END

//BIND_METHOD ImageFloat putpixel
{
  int x,y;
  float pixelvalue;

  LUABIND_CHECK_ARGN(==,3);
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  LUABIND_GET_PARAMETER(3, float, pixelvalue);
  
  // check range
  if ((pixelvalue < 0) || (pixelvalue > 1)) {
    LUABIND_ERROR("pixel value must be in range [0,1]");
  }
  
  obj->operator()(x,y) = pixelvalue;
  return 0;
}
//BIND_END

//BIND_METHOD ImageFloat clone
{
  LUABIND_CHECK_ARGN(==,0);
  ImageFloat *theclone = obj->clone();
  LUABIND_RETURN(ImageFloat, theclone);
}
//BIND_END

//BIND_METHOD ImageFloat projection_h
{
  LUABIND_CHECK_ARGN(==,0);
  Basics::MatrixFloat *mat;
  obj->projection_h(&mat);
  LUABIND_RETURN(MatrixFloat, mat);
}
//BIND_END

//BIND_METHOD ImageFloat projection_v
{
  LUABIND_CHECK_ARGN(==,0);
  Basics::MatrixFloat *mat;
  obj->projection_v(&mat);
  LUABIND_RETURN(MatrixFloat, mat);
}
//BIND_END

//BIND_METHOD ImageFloat shear_h
{
  float angle;
  float default_value;
  const char *units;

  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_GET_PARAMETER(1, float, angle);
  LUABIND_GET_OPTIONAL_PARAMETER(2, string, units, "rad");
  LUABIND_GET_OPTIONAL_PARAMETER(3, float, default_value, CTEBLANCO);

  constString csopt = constString(units);
  if (csopt == "deg") angle = angle/180.0f*M_PI;
  else if (csopt == "grad") angle = angle/200.0f*M_PI;

  ImageFloat *result = obj->shear_h(angle, default_value);

  LUABIND_RETURN(ImageFloat, result);
}
//BIND_END

//BIND_METHOD ImageFloat shear_h_inplace
{
  float angle;
  float default_value;
  const char *units;

  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_GET_PARAMETER(1, float, angle);
  LUABIND_GET_OPTIONAL_PARAMETER(2, string, units, "rad");
  LUABIND_GET_OPTIONAL_PARAMETER(3, float, default_value, CTEBLANCO);

  constString csopt = constString(units);
  if (csopt == "deg") angle = angle/180.0f*M_PI;
  else if (csopt == "grad") angle = angle/200.0f*M_PI;

  obj->shear_h_inplace(angle, default_value);
  return 0;
}
//BIND_END

//BIND_METHOD ImageFloat min_bounding_box
{
  float threshold;

  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, float, threshold);

  int w,h,x,y;
  obj->min_bounding_box(threshold, &w, &h, &x, &y);

  LUABIND_RETURN(int, w);
  LUABIND_RETURN(int, h);
  LUABIND_RETURN(int, x);
  LUABIND_RETURN(int, y);
}
//BIND_END

//BIND_METHOD ImageFloat copy
{
  ImageFloat *src;
  int dst_x, dst_y;

  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_GET_PARAMETER(1, ImageFloat, src);
  LUABIND_GET_PARAMETER(2, int, dst_x);
  LUABIND_GET_PARAMETER(3, int, dst_y);

  obj->copy(src, dst_x, dst_y);

  return 0;
}
//BIND_END

//BIND_METHOD ImageFloat substract
{
  ImageFloat *img;
  
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  ImageFloat *res = obj->substract_image(img, 0.0, 1.0);

  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat threshold
{
  
  LUABIND_CHECK_ARGN(==,2);
  float low, high;

  LUABIND_GET_PARAMETER(1, float, low);
  LUABIND_GET_PARAMETER(2, float, high);
  
  obj->threshold_image(low, high,0.0, 1.0);

  return 0;
}
//BIND_END

//BIND_METHOD ImageFloat rotate90cw
{
  int param; // must be +1 or -1
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, param);
  
  ImageFloat *res = 0;
  
  if (param == 1)
	  res = obj->rotate90_cw();
  else if (param == -1)
	  res = obj->rotate90_ccw();
  else{
    LUABIND_ERROR("the argument of rotate90cw must be +1 or -1");
  }

  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat invert_colors
{
  LUABIND_CHECK_ARGN(==,0);
  
  ImageFloat *res = obj->invert_colors();

  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat remove_blank_columns 
{
  LUABIND_CHECK_ARGN(==,0);

  ImageFloat *res;
  res=obj->remove_blank_columns();

  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat add_rows 
{
  LUABIND_CHECK_ARGN(==,3);
  int top_rows, bottom_rows;
  float value = 0.0;

  LUABIND_GET_PARAMETER(1, int, top_rows);
  LUABIND_GET_PARAMETER(2, int, bottom_rows);

  LUABIND_GET_PARAMETER(3, float, value);
  ImageFloat *res;
  res=obj->add_rows(top_rows, bottom_rows, value);

  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat convolution5x5
{
  LUABIND_CHECK_ARGN(>,0);
  LUABIND_CHECK_ARGN(<,3);
  LUABIND_CHECK_PARAMETER(1, table);
  float kernel[25];
  LUABIND_TABLE_TO_VECTOR(1, float, kernel, 25);

  ImageFloat *res;
  if (lua_gettop(L) == 1)
    res = obj->convolution5x5(kernel);
  else {
    float default_value;
    LUABIND_GET_PARAMETER(2, float, default_value);
    res = obj->convolution5x5(kernel, default_value);
  }

  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat resize
{
  LUABIND_CHECK_ARGN(==, 2);
  int x, y;
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  ImageFloat *res = obj->resize(x,y);
  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat upsample
{
  LUABIND_CHECK_ARGN(==, 2);
  int x, y;
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  ImageFloat *res = obj->upsample(x,y);
  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END

//BIND_METHOD ImageFloat affine_transform
{
  Basics::AffineTransform2D *trans;
  float default_value;
  int offset_x, offset_y;
  ImageFloat *res;

  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_GET_PARAMETER(1, AffineTransform2D, trans);
  LUABIND_GET_PARAMETER(2, float, default_value);
  
  res = obj->affine_transform(trans, default_value, &offset_x, &offset_y);
  
  LUABIND_RETURN(ImageFloat, res);
  LUABIND_RETURN(int, offset_x);
  LUABIND_RETURN(int, offset_y);
}
//BIND_END

//BIND_METHOD ImageFloat to_RGB
{
  LUABIND_CHECK_ARGN(==,0);
  ImageFloatRGB *res = grayscale_to_RGB(obj);
  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END
//BIND_METHOD ImageFloat comb_lineal_forward
{
  LUABIND_CHECK_ARGN(==, 7);
  int x, y, alto, ancho, minialto, miniancho, output_size;
  Basics::LinearCombConfFloat *cl;
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  LUABIND_GET_PARAMETER(3, int, ancho);
  LUABIND_GET_PARAMETER(4, int, alto);
  LUABIND_GET_PARAMETER(5, int, miniancho);
  LUABIND_GET_PARAMETER(6, int, minialto);
  LUABIND_GET_PARAMETER(7, LinearCombConfFloat, cl);

  Basics::MatrixFloat *res = obj->comb_lineal_forward(x,y,ancho, alto,
                                                      miniancho, minialto, cl);
  LUABIND_RETURN(MatrixFloat, res);

}
//BIND_END
