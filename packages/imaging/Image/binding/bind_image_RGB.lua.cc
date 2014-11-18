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
#include "bind_image.h"

using namespace AprilUtils;
using namespace AprilMath;
using namespace Basics;
//BIND_END

//BIND_HEADER_H
#include "utilMatrixFloat.h"
#include "utilImageFloat.h"
#include "bind_matrix.h"
#include "bind_affine_transform.h"
#include <cmath>

using namespace Imaging;
//BIND_END

//BIND_LUACLASSNAME ImageFloatRGB ImageRGB
//BIND_CPP_CLASS ImageFloatRGB

//BIND_CONSTRUCTOR ImageFloatRGB
// este constructor recibe una matrix
{
  LUABIND_CHECK_ARGN(==,1);
  Basics::MatrixFloat      *img;
  LUABIND_GET_PARAMETER(1, MatrixFloat, img);
  if (img->getNumDim() != 3)
    LUABIND_ERROR("Needs a matrix with 3 dimensions");
  if (img->getDimSize(2) != 3)
    LUABIND_ERROR("Needs a matrix with 3 components (R,G,B) at the 3rd dimension");
  if (!img->isSimple())
    LUABIND_ERROR("ImageRGB needs a simple matrix: row_major and contiguous\n");
  GPUMirroredMemoryBlock<FloatRGB> *float_rgb_mem;
  float_rgb_mem = img->getRawDataAccess()->reinterpretAs<FloatRGB>();
  int dims[2] = { img->getDimSize(0), img->getDimSize(1) };
  Basics::Matrix<FloatRGB> *img_rgb = new Basics::Matrix<FloatRGB>(2, dims,
                                                                   float_rgb_mem);
  //
  obj = new ImageFloatRGB(img_rgb);
  LUABIND_RETURN(ImageFloatRGB, obj);
}
//BIND_END

//BIND_FUNCTION ImageRGB.empty
{
    int w,h;
    LUABIND_CHECK_ARGN(==,2);
    LUABIND_GET_PARAMETER(1, int, w);
    LUABIND_GET_PARAMETER(2, int, h);

    Basics::Matrix<FloatRGB> *m = new Basics::Matrix<FloatRGB>(2, w, h);
    ImageFloatRGB *result = new ImageFloatRGB(m);
    
    LUABIND_RETURN(ImageFloatRGB, result);
}
//BIND_END


//BIND_DESTRUCTOR ImageFloatRGB
{
}
//BIND_END

//BIND_METHOD ImageFloatRGB crop
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
  ImageFloatRGB *cropimage = obj->crop(width,height,offsetx,offsety);
  
  LUABIND_RETURN(ImageFloatRGB, cropimage);
}
//BIND_END

//BIND_METHOD ImageFloatRGB info
{
  fprintf(stderr, "WARNING: ImageFloatRGB.info() is deprecated\n"
                  "         Use matrix() or geometry() as needed.\n");
  LUABIND_CHECK_ARGN(==,0);
  
  LUABIND_RETURN(string, "");
  LUABIND_RETURN(int, obj->width());
  LUABIND_RETURN(int, obj->height());
  LUABIND_RETURN(int, obj->offset_width());
  LUABIND_RETURN(int, obj->offset_height());
}
//BIND_END

//BIND_METHOD ImageFloatRGB matrix
{
  LUABIND_CHECK_ARGN(==,0);
  Basics::Matrix<FloatRGB> *img_rgb = obj->getMatrix();
  AprilMath::GPUMirroredMemoryBlock<float> *mat_mem;
  mat_mem = img_rgb->getRawDataAccess()->reinterpretAs<float>();
  int dims[3] = { img_rgb->getDimSize(0), img_rgb->getDimSize(1), 3 };
  Basics::MatrixFloat *output = new Basics::MatrixFloat(3, dims,
                                                        mat_mem);
  LUABIND_RETURN(MatrixFloat, output);
}
//BIND_END

//BIND_METHOD ImageFloatRGB geometry
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(int, obj->width());
  LUABIND_RETURN(int, obj->height());
  LUABIND_RETURN(int, obj->offset_width());
  LUABIND_RETURN(int, obj->offset_height());
}
//BIND_END

//BIND_METHOD ImageFloatRGB getpixel
{
  int x,y;

  LUABIND_CHECK_ARGN(==,2);
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  
  FloatRGB rgb = obj->operator()(x,y);
  
  LUABIND_RETURN(float, rgb.r);
  LUABIND_RETURN(float, rgb.g);
  LUABIND_RETURN(float, rgb.b);
}
//BIND_END

//BIND_METHOD ImageFloatRGB putpixel
{
  int x,y;
  float r, g, b;

  LUABIND_CHECK_ARGN(==,5);
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  LUABIND_GET_PARAMETER(3, float, r);
  LUABIND_GET_PARAMETER(4, float, g);
  LUABIND_GET_PARAMETER(5, float, b);
  
  obj->operator()(x,y) = FloatRGB(r,g,b);
  return 0;
}
//BIND_END

//BIND_METHOD ImageFloatRGB clone
{
  LUABIND_CHECK_ARGN(==,0);
  ImageFloatRGB *theclone = obj->clone();
  LUABIND_RETURN(ImageFloatRGB, theclone);
}
//BIND_END

//BIND_METHOD ImageFloatRGB shear_h
{
  float angle;
  float default_r, default_g, default_b;
  const char *units;

  LUABIND_CHECK_ARGN(>=,4);
  LUABIND_GET_PARAMETER(1, float, angle);
  LUABIND_GET_PARAMETER(2, float, default_r);
  LUABIND_GET_PARAMETER(3, float, default_g);
  LUABIND_GET_PARAMETER(4, float, default_b);
  LUABIND_GET_OPTIONAL_PARAMETER(5, string, units, "rad");

  constString csopt = constString(units);
  if (csopt == "deg") angle = angle/180.0f*M_PI;
  else if (csopt == "grad") angle = angle/200.0f*M_PI;

  ImageFloatRGB *result = obj->shear_h(angle, FloatRGB(default_r, default_g, default_b));

  LUABIND_RETURN(ImageFloatRGB, result);
}
//BIND_END

//BIND_METHOD ImageFloatRGB shear_h_inplace
{
  float angle;
  float default_r, default_g, default_b;
  const char *units;

  LUABIND_CHECK_ARGN(>=,4);
  LUABIND_GET_PARAMETER(1, float, angle);
  LUABIND_GET_PARAMETER(2, float, default_r);
  LUABIND_GET_PARAMETER(3, float, default_g);
  LUABIND_GET_PARAMETER(4, float, default_b);
  LUABIND_GET_OPTIONAL_PARAMETER(5, string, units, "rad");

  constString csopt = constString(units);
  if (csopt == "deg") angle = angle/180.0f*M_PI;
  else if (csopt == "grad") angle = angle/200.0f*M_PI;

  obj->shear_h_inplace(angle, FloatRGB(default_r, default_g, default_b));
  return 0;
}
//BIND_END

//BIND_METHOD ImageFloatRGB copy
{
  ImageFloatRGB *src;
  int dst_x, dst_y;

  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_GET_PARAMETER(1, ImageFloatRGB, src);
  LUABIND_GET_PARAMETER(2, int, dst_x);
  LUABIND_GET_PARAMETER(3, int, dst_y);

  obj->copy(src, dst_x, dst_y);

  return 0;
}
//BIND_END



//BIND_METHOD ImageFloatRGB rotate90cw
{
  int param; // must be +1 or -1
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, param);
  
  ImageFloatRGB *res = 0;
  
  if (param == 1)
	  res = obj->rotate90_cw();
  else if (param == -1)
	  res = obj->rotate90_ccw();
  else{
    LUABIND_ERROR("the argument of rotate90cw must be +1 or -1");
  }

  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END

//BIND_METHOD ImageFloatRGB invert_colors
{
  LUABIND_CHECK_ARGN(==,0);
  
  ImageFloatRGB *res = obj->invert_colors();

  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END

//BIND_METHOD ImageFloatRGB convolution5x5
{
  LUABIND_CHECK_ARGN(>,0);
  LUABIND_CHECK_ARGN(<,5);
  LUABIND_CHECK_PARAMETER(1, table);
  float kernel[25];
  LUABIND_TABLE_TO_VECTOR(1, float, kernel, 25);

  ImageFloatRGB *res;
  if (lua_gettop(L) == 1)
    res = obj->convolution5x5(kernel);
  else {
    float default_r, default_g, default_b;
    LUABIND_GET_PARAMETER(2, float, default_r);
    LUABIND_GET_PARAMETER(3, float, default_g);
    LUABIND_GET_PARAMETER(4, float, default_b);
    res = obj->convolution5x5(kernel, FloatRGB(default_r, default_g, default_b));
  }

  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END

//BIND_METHOD ImageFloatRGB resize
{
  LUABIND_CHECK_ARGN(==, 2);
  int x, y;
  LUABIND_GET_PARAMETER(1, int, x);
  LUABIND_GET_PARAMETER(2, int, y);
  ImageFloatRGB *res = obj->resize(x,y);
  LUABIND_RETURN(ImageFloatRGB, res);
}
//BIND_END

//BIND_METHOD ImageFloatRGB affine_transform
{
  Basics::AffineTransform2D *trans;
  float default_r, default_g, default_b;
  int offset_x, offset_y;
  ImageFloatRGB *res;

  LUABIND_CHECK_ARGN(==, 4);
  LUABIND_GET_PARAMETER(1, AffineTransform2D, trans);
  LUABIND_GET_PARAMETER(2, float, default_r);
  LUABIND_GET_PARAMETER(3, float, default_g);
  LUABIND_GET_PARAMETER(4, float, default_b);
  
  res = obj->affine_transform(trans, FloatRGB(default_r, default_g, default_b), &offset_x, &offset_y);
  
  LUABIND_RETURN(ImageFloatRGB, res);
  LUABIND_RETURN(int, offset_x);
  LUABIND_RETURN(int, offset_y);
}
//BIND_END


//BIND_METHOD ImageFloatRGB to_grayscale
{
  LUABIND_CHECK_ARGN(==,0);
  ImageFloat *res = RGB_to_grayscale(obj);
  LUABIND_RETURN(ImageFloat, res);
}
//BIND_END




