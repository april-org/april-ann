/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera
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
extern "C" {
#include <ctype.h>
}
#include "ignore_result.h"
#include "luabindutil.h"
#include "luabindmacros.h"
#include "matrix_ext.h"
#include "smart_ptr.h"
#include "utilMatrixChar.h"
#include "utilMatrixComplexF.h"
#include "utilMatrixFloat.h"
#include "utilMatrixInt32.h"
//BIND_END

//BIND_HEADER_H
#include "bind_matrix.h"
#include "bind_matrix_bool.h"
#include "bind_matrix_char.h"
#include "bind_matrix_complex_float.h"
#include "bind_matrix_double.h"
#include "bind_matrix_int32.h"
//BIND_END

//BIND_LUACLASSNAME MatrixFloat matrix
//BIND_LUACLASSNAME MatrixBool matrixBool
//BIND_LUACLASSNAME MatrixInt32 matrixInt32
//BIND_LUACLASSNAME MatrixComplexF matrixComplex
//BIND_LUACLASSNAME MatrixChar matrixChar

//////////////////////////////////////////////////////////////////////////////

//BIND_FUNCTION matrix.ext.convolution
{
  MatrixFloat *obj, *kernel, *result; //, *unrolled_kernel, *unrolled_self;
  AprilUtils::UniquePtr<int []> step;
  int D;
  
  LUABIND_CHECK_ARGN(>=, 2);
  LUABIND_CHECK_ARGN(<=, 3);
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_CHECK_PARAMETER(2, table);
  LUABIND_GET_TABLE_PARAMETER(2, D, int, D);
  LUABIND_GET_TABLE_PARAMETER(2, kernel, MatrixFloat, kernel);
  /*
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, unrolled_kernel,
    MatrixFloat, unrolled_kernel, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, unrolled_self,
    MatrixFloat, unrolled_self, 0);
  */
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixFloat, result, 0);
  lua_getfield(L, 2, "step");
  if (!lua_isnil(L, -1)) {
    step = new int[D];
    int len;
    LUABIND_TABLE_GETN(-1, len);
    if (len != D) {
      LUABIND_FERROR2("Incorrect length of step table, found %d, expected %d",
                      len, D);
    }
    LUABIND_TABLE_TO_VECTOR(-1, int, step.get(), D);
  }
  lua_pop(L, 1);
  
  LUABIND_RETURN(MatrixFloat,
                 AprilMath::MatrixExt::Misc::
                 matConvolution(obj, D, step.get(), kernel, result));
  //&unrolled_kernel, &unrolled_self);
  /*LUABIND_RETURN(MatrixFloat, unrolled_kernel);
    LUABIND_RETURN(MatrixFloat, unrolled_self);*/
}
//BIND_END

//BIND_FUNCTION matrix.ext.real_fftwh
{
  MatrixFloat *obj, *dest;
  int wsize, wadvance;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_GET_OPTIONAL_PARAMETER(2, int, wsize, obj->size());
  LUABIND_GET_OPTIONAL_PARAMETER(3, int, wadvance, wsize);
  LUABIND_GET_OPTIONAL_PARAMETER(4, MatrixFloat, dest, 0);
  LUABIND_RETURN(MatrixFloat,
                 AprilMath::MatrixExt::Misc::
		 matRealFFTwithHamming(obj, wsize, wadvance, dest));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_CLASS_METHOD MatrixFloat fromPNM
//DOC_BEGIN
// matrix *fromPNM(string pnm_image)
/// constructor con un argumento que es una cadena con una imagen en
/// formato de netpbm P5 o P6 (binario PGM o PNM)
///@param pnm_image String que contiene la imagen.
//DOC_END
// TODO: poder forzar niveles de gris o color, poder leer PBM
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  LUABIND_CHECK_PARAMETER(1, string);
  bool forcecolor=false,forcegray=false;
  AprilUtils::constString cs,csopt;
  LUABIND_GET_PARAMETER(1,constString,cs);
  LUABIND_GET_OPTIONAL_PARAMETER(2,constString,csopt,
                                 AprilUtils::constString());
  if (csopt == "color") forcecolor = true;
  if (csopt == "gray")  forcegray  = true;
  MatrixFloat *obj;
  if ((obj = readMatrixFloatPNM(cs,forcecolor,forcegray))== 0)
    LUABIND_ERROR("bad format");
  else LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat fromHEX
//DOC_BEGIN
// matrix *fromHEX(width, height, string hex_image)
/// constructor con 3 argumentos que es una cadena con una imagen en
/// escala de grises, 2 caracteres hexadecimales por pixel
///@param width
///@param height
///@param hex_image
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(1, string);
  int width,height;
  AprilUtils::constString cs;
  LUABIND_GET_PARAMETER(1,int,width);
  LUABIND_GET_PARAMETER(2,int,height);
  LUABIND_GET_PARAMETER(3,constString,cs);
  MatrixFloat *obj;
  obj = readMatrixFloatHEX(width,height,cs);
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_METHOD MatrixFloat toHEX
//DOC_BEGIN
// string toHEX()
//DOC_END
{
  char *buffer;
  int   width, height;
  int   longitud = saveMatrixFloatHEX(obj,&buffer, &width, &height);
  if (!buffer) {
    LUABIND_ERROR("bad format");
  }
  LUABIND_RETURN(int, width);
  LUABIND_RETURN(int, height);
  lua_pushlstring(L,buffer,longitud);
  delete[] buffer;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD MatrixFloat toPNM
//DOC_BEGIN
// string toPNM()
/// Devuelve una cadena correspondiente a un fichero PNM (P5 o P6).  La
/// matriz debe ser de dimension 2 o, si es de dimension 3, la tercera
/// dimension debe tener 3 componentes correspondientes respectivamente
/// a los colores RGB. El 0 se interpreta como negro, el 1 como blanco
/// y saturan (es decir, un -1 es como 0 y un 5 es como 1).
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  char *buffer;
  int longitud = saveMatrixFloatPNM(obj,&buffer);
  if (!buffer)
    LUABIND_ERROR("bad format");
  lua_pushlstring(L,buffer,longitud);
  delete[] buffer;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_METHOD MatrixBool count_ones
{
  int count=0;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end(); ++it) {
    if (*it) ++count;
  }
  LUABIND_RETURN(int, count);
}
//BIND_END

//BIND_METHOD MatrixBool count_zeros
{
  int count=0;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end(); ++it) {
    if (!(*it)) ++count;
  }
  LUABIND_RETURN(int, count);
}
//BIND_END

//BIND_METHOD MatrixBool any
{
  bool result = false;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end() && !result; ++it) {
    result = result || (*it);
  }
  LUABIND_RETURN(boolean, result);
}
//BIND_END

//BIND_METHOD MatrixBool all
{
  bool result = true;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end() && result; ++it) {
    result = result && (*it);
  }
  LUABIND_RETURN(boolean, result);
}
//BIND_END

//BIND_METHOD MatrixBool complement
{
  for (MatrixBool::iterator it(obj->begin());
       it != obj->end(); ++it) {
    if (*it) *it = false;
    else *it = true;
  }
  LUABIND_RETURN(MatrixBool, obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_METHOD MatrixComplexF to_float
{
  LUABIND_RETURN(MatrixFloat, convertFromMatrixComplexFToMatrixFloat(obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF conj
{
  applyConjugateInPlace(obj);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF real
{
  LUABIND_RETURN(MatrixFloat, realPartFromMatrixComplexFToMatrixFloat(obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF img
{
  LUABIND_RETURN(MatrixFloat, imgPartFromMatrixComplexFToMatrixFloat(obj));
}
//BIND_END


//BIND_METHOD MatrixComplexF abs
{
  LUABIND_RETURN(MatrixFloat, absFromMatrixComplexFToMatrixFloat(obj));
}
//BIND_END

//BIND_METHOD MatrixComplexF angle
{
  LUABIND_RETURN(MatrixFloat, angleFromMatrixComplexFToMatrixFloat(obj));
}
//BIND_END
