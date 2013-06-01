/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include <cmath> // para isfinite
//BIND_END

//BIND_HEADER_H
#include "utilMatrixFloat.h"
#include "utilLua.h"
#include <cmath> // para isfinite
//BIND_END

//BIND_LUACLASSNAME MatrixFloat matrix
//BIND_CPP_CLASS MatrixFloat

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// matrix(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  int i,argn;
  argn = lua_gettop(L); // number of arguments
  LUABIND_CHECK_ARGN(>=, 1);
  int ndims = (lua_istable(L,argn)) ? argn-1 : argn;
  int *dim;
  if (ndims == 0) { // caso matrix{valores}
    ndims = 1;
    dim = new int[ndims];
    LUABIND_TABLE_GETN(1, dim[0]);
  } else {
    dim = new int[ndims];
    for (i=1; i <= ndims; i++) {
      if (!lua_isnumber(L,i))
	// TODO: Este mensaje de error parece que no es correcto... y no se todavia por que!!!
	LUABIND_FERROR2("incorrect argument to matrix dimension (arg %d must"
			" be a number and is a %s)",
			i, lua_typename(L,i));
      dim[i-1] = (int)lua_tonumber(L,i);
      if (dim[i-1] <= 0)
	LUABIND_FERROR1("incorrect argument to matrix dimension (arg %d must be >0)",i);
    }
  }
  MatrixFloat* obj = new MatrixFloat(ndims,dim);
  if (lua_istable(L,argn)) {
    int talla = obj->size;
    float *data=obj->getData();
    for (int i=1; i <= talla; i++) {
      lua_rawgeti(L,argn,i);
      data[i-1] = (float)luaL_checknumber(L, -1);
      lua_remove(L,-1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_DESTRUCTOR MatrixFloat
{
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat fromFilename
//DOC_BEGIN
// matrix *fromFilename(string filename)
/// Constructor con un argumento que es un fichero que contiene la matriz.  Pueden haber
/// comentarios que son lineas que empiezan con un simbolo '#'.  La
/// primera linea contiene tantos valores numericos como dimensiones
/// tenga la matriz y que corresponde al numero de componentes en cada
/// dimension.  La siguiente linea contiene la palabra "ascii" o
/// "binary".  El resto de lineas contienen los datos propiamente
/// dichos.
///@param filename Es un string que indica el nombre del fichero.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  const char *filename;
  LUABIND_GET_PARAMETER(1,string,filename);
  MatrixFloat *obj;
  ReadFileStream f(filename);
  if ((obj = readMatrixFloatFromStream(f)) == 0)
    LUABIND_ERROR("bad format");
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END


//BIND_CLASS_METHOD MatrixFloat fromString
//DOC_BEGIN
// matrix *fromString(string description)
/// Constructor con un argumento que es una cadena.  Pueden haber
/// comentarios que son lineas que empiezan con un simbolo '#'.  La
/// primera linea contiene tantos valores numericos como dimensiones
/// tenga la matriz y que corresponde al numero de componentes en cada
/// dimension.  La siguiente linea contiene la palabra "ascii" o
/// "binary".  El resto de lineas contienen los datos propiamente
/// dichos.
///@param description Es un string que describe a la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  constString cs;
  LUABIND_GET_PARAMETER(1,constString,cs);
  MatrixFloat *obj;
  if ((obj = readMatrixFloatFromStream(cs)) == 0)
    LUABIND_ERROR("bad format");
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_METHOD MatrixFloat toFilename
//DOC_BEGIN
// string toFilename(string filename, string type='ascii')
/// Permite salvar una matriz con un formato tal y como se carga con el
/// metodo fromString. El unico argumento opcional indica el tipo 'ascii'
/// o 'binary'.
///@param filename Indica el nombre del fichero.
///@param type Parametro opcional. Puede ser 'ascii' o 'binary', y por defecto es 'ascii'.
//DOC_END
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  const char *filename;
  constString cs;
  LUABIND_GET_PARAMETER(1, string, filename);
  LUABIND_GET_OPTIONAL_PARAMETER(2,constString,cs,constString("ascii"));
  bool is_ascii = (cs == "ascii");
  FILE *f = fopen(filename, "w");
  saveMatrixFloatToFile(obj,f,is_ascii);
  fclose(f);
}
//BIND_END

//BIND_METHOD MatrixFloat toString
//DOC_BEGIN
// string toString(string type='ascii')
/// Permite salvar una matriz con un formato tal y como se carga con el
/// metodo fromString. El unico argumento opcional indica el tipo 'ascii'
/// o 'binary'.
///@param type ParÃ¡metro opcional. Puede ser 'ascii' o 'binary', y por defecto es 'ascii'.
//DOC_END
{
  LUABIND_CHECK_ARGN(<=, 1);
  constString cs;
  LUABIND_GET_OPTIONAL_PARAMETER(1,constString,cs,constString("ascii"));
  bool is_ascii = (cs == "ascii");
  char *buffer;
  int longitud = saveMatrixFloatToString(obj,&buffer,is_ascii);
  lua_pushlstring(L,buffer,longitud);
  delete[] buffer;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

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
  constString cs,csopt;
  LUABIND_GET_PARAMETER(1,constString,cs);
  LUABIND_GET_OPTIONAL_PARAMETER(2,constString,csopt,constString());
  if (csopt == "color") forcecolor = true;
  if (csopt == "gray")  forcegray  = true;
  MatrixFloat *obj;
  if ((obj = readMatrixFloatPNM(cs,forcecolor,forcegray))== 0)
    LUABIND_ERROR("bad format");
  LUABIND_RETURN(MatrixFloat,obj);
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
  constString cs;
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
  if (!buffer)
    LUABIND_ERROR("bad format");
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

//BIND_METHOD MatrixFloat set
//DOC_BEGIN
// void set(table matrix_values)
/// Permite dar valores a una matriz. Require una tabla con un numero
/// de argumentos igual al numero de elementos de la matriz.
///@param matrix_values Tabla con los elementos de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int veclen;
  LUABIND_TABLE_GETN(1, veclen);
  if (veclen != obj->size)
    LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size);
  LUABIND_TABLE_TO_VECTOR(1, float, obj->getData(), veclen);
}
//BIND_END

//BIND_METHOD MatrixFloat getElement
//DOC_BEGIN
// float get_element(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->numDim)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->numDim);

  float ret;
  if (obj->numDim == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->matrixSize[0]) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->matrixSize[0]);
    }
    ret = (*obj)(v1-1);
  }
  else if (obj->numDim == 2) {
    int v1, v2;
    LUABIND_GET_PARAMETER(1,int,v1);
    LUABIND_GET_PARAMETER(2,int,v2);
    if (v1<1 || v1 > obj->matrixSize[0]) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->matrixSize[0]);
    }
    if (v2<1 || v2 > obj->matrixSize[1]) {
      LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
		      v2, obj->matrixSize[1]);
    }
    ret = (*obj)(v1-1, v2-1);
  }
  else {
    int value;
    LUABIND_GET_PARAMETER(1,int,value);
    if (value<1 || value > obj->matrixSize[0]) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      value, obj->matrixSize[0]);
    }
    int rawpos = value-1;
    for (int i=1; i<obj->numDim; ++i) {
      LUABIND_GET_PARAMETER(i+1,int,value);
      if (value<1 || value > obj->matrixSize[i]) {
	LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
			value, obj->matrixSize[i]);
      }
      rawpos = rawpos*obj->matrixSize[i] + (value-1);
    }
    ret = obj->data->get(rawpos);
  }
  LUABIND_RETURN(float, ret);
}
//BIND_END

//BIND_METHOD MatrixFloat setElement
//DOC_BEGIN
// float set_element(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->numDim+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->numDim+1);
  float f;
  if (obj->numDim == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->matrixSize[0]) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->matrixSize[0]);
    }
    LUABIND_GET_PARAMETER(obj->numDim+1,float,f);
    (*obj)(v1-1) = f;
  }
  else if (obj->numDim == 2) {
    int v1, v2;
    LUABIND_GET_PARAMETER(1,int,v1);
    LUABIND_GET_PARAMETER(2,int,v2);
    if (v1<1 || v1 > obj->matrixSize[0]) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->matrixSize[0]);
    }
    if (v2<1 || v2 > obj->matrixSize[1]) {
      LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
		      v2, obj->matrixSize[1]);
    }
    LUABIND_GET_PARAMETER(obj->numDim+1,float,f);
    (*obj)(v1-1, v2-1) = f;
  }
  else {
    int value;
    LUABIND_GET_PARAMETER(1,int,value);
    if (value<1 || value > obj->matrixSize[0]) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      value, obj->matrixSize[0]);
    }
    int rawpos = value-1;
    for (int i=1; i<obj->numDim; ++i) {
      LUABIND_GET_PARAMETER(i+1,int,value);
      if (value<1 || value > obj->matrixSize[i]) {
	LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
			value, obj->matrixSize[i]);
      }
      rawpos = rawpos*obj->matrixSize[i] + (value-1);
    }
    float f;
    LUABIND_GET_PARAMETER(obj->numDim+1,float,f);
    obj->data->get(rawpos) = f;
  }
}
//BIND_END

//BIND_METHOD MatrixFloat fill
//DOC_BEGIN
// void fill(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, float);
  float value,*data=obj->getData();
  LUABIND_GET_PARAMETER(1,float,value);
  int talla = obj->size;
  for (int i=0; i < talla; i++)
    data[i] = value;
}
//BIND_END

//BIND_METHOD MatrixFloat dim
//DOC_BEGIN
// table dim()
/// Devuelve una tabla con las dimensiones de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  int  ndim=obj->numDim;
  int *d   =obj->matrixSize;
  LUABIND_VECTOR_TO_NEW_TABLE(int, d, ndim);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD MatrixFloat clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  MatrixFloat *obj2 = obj->clone();
  LUABIND_RETURN(MatrixFloat,obj2);
}
//BIND_END

//BIND_METHOD MatrixFloat isfinite
//DOC_BEGIN
// bool isfinite
/// Devuelve false si algun valor es nan o infinito.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  int resul = 1;
  for (int i=0; resul && i<obj->size; i++)
    //if (!isfinite(obj->data[i])) resul = 0;
    if ((obj->data->get(i) - obj->data->get(i)) != 0.0f) resul = 0;
  LUABIND_RETURN(boolean,resul);
}
//BIND_END

//BIND_METHOD MatrixFloat adjust_range
//DOC_BEGIN
// void adjust_range(float min, float max)
/// Ajusta el rango de valores de la matriz para que estÃ© en [min,
/// max].
//DOC_END
{
  float rmin,rmax;
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, float);
  LUABIND_CHECK_PARAMETER(2, float);
  LUABIND_GET_PARAMETER(1,float,rmin);
  LUABIND_GET_PARAMETER(2,float,rmax);

  // ajusta los valores a un rango predeterminado
  float mmin = obj->data->get(0);
  float mmax = obj->data->get(0);
  for (int i=0; i<obj->size; i++) {
    if (mmin > obj->data->get(i))
      mmin = obj->data->get(i);
    if (mmax < obj->data->get(i))
      mmax = obj->data->get(i);
  }
  if (mmax - mmin == 0) {
    // caso especial, poner todos al valor inferior
    for (int i=0; i<obj->size; i++) {
      obj->data->get(i) = rmin;
    }
  } else {
    float offset = rmin-mmin;
    double ratio = (rmax-rmin)/(mmax-mmin);
    for (int i=0; i<obj->size; i++) {
      obj->data->get(i) = ratio*(obj->data->get(i)+offset);
    }
  }
}
//BIND_END

//BIND_METHOD MatrixFloat toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_VECTOR_TO_NEW_TABLE(float, obj->getData(), obj->size);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD MatrixFloat getMinValue
{
  float min = obj->data->get(0);
  for (int i=1; i<obj->size; ++i) {
    if (obj->data->get(i) < min)
      min = obj->data->get(i);
  }
  LUABIND_RETURN(float, min);
}
//BIND_END

//BIND_METHOD MatrixFloat getMaxValue
{
  float max = obj->data->get(0);
  for (int i=1; i<obj->size; ++i) {
    if (obj->data->get(i) > max)
      max = obj->data->get(i);
  }
  LUABIND_RETURN(float, max);
}
//BIND_END

//BIND_METHOD MatrixFloat clamp
{
  LUABIND_CHECK_ARGN(==, 2);
  float lower,upper;
  LUABIND_GET_PARAMETER(1, float, lower);
  LUABIND_GET_PARAMETER(2, float, upper);
  obj->clamp(lower,upper);
}
//BIND_END

//BIND_METHOD MatrixFloat addition
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat,*resul;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  if (!obj->sameDim(mat))
    LUABIND_ERROR("matrix addition wrong dimensions");
  resul = obj->addition(mat);
  LUABIND_RETURN(MatrixFloat, resul);
}
//BIND_END

//BIND_METHOD MatrixFloat accumulate_addition
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  if (!obj->sameDim(mat))
    LUABIND_ERROR("matrix accumulate_addition wrong dimensions");
  obj->accumulate_addition(mat);
}
//BIND_END

//BIND_METHOD MatrixFloat substraction
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat,*resul;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  if (!obj->sameDim(mat))
    LUABIND_ERROR("matrix substraction wrong dimensions");
  resul = obj->substraction(mat);
  LUABIND_RETURN(MatrixFloat, resul);
}
//BIND_END

//BIND_METHOD MatrixFloat accumulate_substraction
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  if (!obj->sameDim(mat))
    LUABIND_ERROR("matrix accumulate_substraction wrong dimensions");
  obj->accumulate_substraction(mat);
}
//BIND_END

//BIND_METHOD MatrixFloat multiply
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat,*resul;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  resul = obj->multiply(mat);
  if (resul == 0)
    LUABIND_ERROR("matrix multiply wrong dimensions");
  LUABIND_RETURN(MatrixFloat, resul);
}
//BIND_END

//BIND_METHOD MatrixFloat multiply_by_scalar
{
  LUABIND_CHECK_ARGN(==, 1);
  float value;
  LUABIND_GET_PARAMETER(1, float, value);
  obj->multiply_by_scalar(value);
}
//BIND_END

