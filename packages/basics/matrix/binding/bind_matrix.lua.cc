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
#include "bind_mtrand.h"
#include <cmath> // para isfinite
#include "luabindutil.h"
#include "luabindmacros.h"

#define FUNCTION_NAME "read_vector"
int *read_vector(lua_State *L, const char *key, int num_dim, int add) {
  int *v=0;
  lua_getfield(L, 1, key);
  if (!lua_isnil(L, -1)) {
    LUABIND_CHECK_PARAMETER(-1, table);
    int table_len;
    LUABIND_TABLE_GETN(-1, table_len);
    if (table_len != num_dim)
      LUABIND_FERROR3("Table '%s' with incorrect size, expected %d, found %d",
		      key, num_dim, table_len);
    v = new int[num_dim];
    for(int i=0; i < num_dim; i++) {
      lua_rawgeti(L, -1, i+1);
      v[i] = static_cast<int>(lua_tonumber(L, -1)) + add;
      lua_pop(L,1);
    }
  }
  lua_pop(L, 1);
  return v;
}
#undef FUNCTION_NAME

int sliding_window_iterator_function(lua_State *L) {
  SlidingWindow *obj = lua_toSlidingWindow(L,1);
  if (obj->isEnd()) {
    lua_pushnil(L);
    return 1;
  }
  // lua_pushSlidingWindow(L, obj);
  MatrixFloat *mat = obj->getMatrix();
  lua_pushMatrixFloat(L, mat);
  obj->next();
  return 1;
}

//BIND_END

//BIND_HEADER_H
#include "utilMatrixFloat.h"
#include "utilLua.h"
#include <cmath> // para isfinite
typedef MatrixFloat::sliding_window SlidingWindow;
//BIND_END

//BIND_LUACLASSNAME MatrixFloat matrix
//BIND_CPP_CLASS MatrixFloat

//BIND_LUACLASSNAME SlidingWindow matrix.__sliding_window__
//BIND_CPP_CLASS SlidingWindow

//BIND_CONSTRUCTOR SlidingWindow
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindow get_matrix
{
  bool clone;
  LUABIND_GET_OPTIONAL_PARAMETER(1, bool, clone, false);
  LUABIND_RETURN(MatrixFloat, obj->getMatrix(clone));
}
//BIND_END

//BIND_METHOD SlidingWindow next
{
  LUABIND_RETURN(SlidingWindow, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindow set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindow, obj);
}
//BIND_END

//BIND_METHOD SlidingWindow num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindow coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindow is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindow iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_iterator_function);
  LUABIND_RETURN(SlidingWindow,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

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
  int ndims = (!lua_isnumber(L,argn)) ? argn-1 : argn;
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
  MatrixFloat* obj;
  obj = new MatrixFloat(ndims,dim);
  if (lua_istable(L,argn)) {
    int i=1;
    for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it, ++i) {
      lua_rawgeti(L,argn,i);
      *it = (float)luaL_checknumber(L, -1);
      lua_remove(L,-1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat col_major
//DOC_BEGIN
// col_major_matrix(int dim1, int dim2, ..., table mat=nil)
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
  int ndims = (!lua_isnumber(L,argn)) ? argn-1 : argn;
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
  MatrixFloat* obj;
  obj = new MatrixFloat(ndims,dim,CblasColMajor);
  if (lua_istable(L,argn)) {
    int i=1;
    for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it, ++i) {
      lua_rawgeti(L,argn,i);
      *it = (float)luaL_checknumber(L, -1);
      lua_remove(L,-1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_METHOD MatrixFloat size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixFloat rewrap
{
  LUABIND_CHECK_ARGN(>=, 1);
  int ndims;
  ndims = lua_gettop(L); // number of dimensions
  int *dims = new int[ndims];
  for (int i=1; i <= ndims; i++) {
    LUABIND_GET_PARAMETER(i, int, dims[i-1]);
    if (dims[i-1] <= 0)
      LUABIND_FERROR1("incorrect argument to matrix dimension (arg %d must be >0)",i);
  }
  MatrixFloat *new_obj = obj->rewrap(dims, ndims);
  delete[] dims;
  LUABIND_RETURN(MatrixFloat,new_obj);
}
//BIND_END

//BIND_METHOD MatrixFloat get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p",
	  (void*)obj,
	  (void*)obj->getRawDataAccess());
  LUABIND_RETURN(string, buff);
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
  else LUABIND_RETURN(MatrixFloat,obj);
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
  else LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_METHOD MatrixFloat toFilename
//DOC_BEGIN
// void toFilename(string filename, string type='ascii')
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

//BIND_METHOD MatrixFloat copy_from_table
//DOC_BEGIN
// void copy_from_table(table matrix_values)
/// Permite dar valores a una matriz. Require una tabla con un numero
/// de argumentos igual al numero de elementos de la matriz.
///@param matrix_values Tabla con los elementos de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int veclen;
  LUABIND_TABLE_GETN(1, veclen);
  if (veclen != obj->size())
    LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size());
  int i=1;
  for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it, ++i) {
    lua_rawgeti(L,1,i);
    *it = (float)luaL_checknumber(L, -1);
    lua_remove(L,-1);
  }
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat get
//DOC_BEGIN
// float get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  float ret;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    ret = (*obj)(v1-1);
  }
  else if (obj->getNumDim() == 2) {
    int v1, v2;
    LUABIND_GET_PARAMETER(1,int,v1);
    LUABIND_GET_PARAMETER(2,int,v2);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    if (v2<1 || v2 > obj->getDimSize(1)) {
      LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
		      v2, obj->getDimSize(1));
    }
    ret = (*obj)(v1-1, v2-1);
  }
  else {
    int *coords = new int[obj->getNumDim()];
    for (int i=0; i<obj->getNumDim(); ++i) {
      LUABIND_GET_PARAMETER(i+1,int,coords[i]);
      if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
	LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
			coords[i], obj->getDimSize(i));
      }
      coords[i]--;
    }
    ret = (*obj)(coords, obj->getNumDim());
    delete[] coords;
  }
  LUABIND_RETURN(float, ret);
}
//BIND_END

//BIND_METHOD MatrixFloat set
//DOC_BEGIN
// float set(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim()+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+1);
  float f;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,float,f);
    (*obj)(v1-1) = f;
  }
  else if (obj->getNumDim() == 2) {
    int v1, v2;
    LUABIND_GET_PARAMETER(1,int,v1);
    LUABIND_GET_PARAMETER(2,int,v2);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    if (v2<1 || v2 > obj->getDimSize(1)) {
      LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
		      v2, obj->getDimSize(1));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,float,f);
    (*obj)(v1-1, v2-1) = f;
  }
  else {
    int *coords = new int[obj->getNumDim()];
    for (int i=0; i<obj->getNumDim(); ++i) {
      LUABIND_GET_PARAMETER(i+1,int,coords[i]);
      if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
	LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
			coords[i], obj->getDimSize(i));
      }
      coords[i]--;
    }
    float f;
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,float,f);
    (*obj)(coords, obj->getNumDim()) = f;
    delete[] coords;
  }
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixFloat raw_get
{
  int raw_pos;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_RETURN(float, (*obj)[raw_pos]);
}
//BIND_END

//BIND_METHOD MatrixFloat raw_set
{
  int raw_pos;
  float value;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_GET_PARAMETER(2, float, value);
  (*obj)[raw_pos] = value;
  LUABIND_RETURN(MatrixFloat, obj);
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
  float value;
  LUABIND_GET_PARAMETER(1,float,value);
  obj->fill(value);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat zeros
//DOC_BEGIN
// void zeros(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  obj->zeros();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat ones
//DOC_BEGIN
// void onex(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  obj->ones();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat set_use_cuda
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, bool);
  bool v;
  LUABIND_GET_PARAMETER(1,bool, v);
  obj->setUseCuda(v);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat get_major_order
{
  if (obj->getMajorOrder() == CblasRowMajor)
    LUABIND_RETURN(string, "row_major");
  else LUABIND_RETURN(string, "col_major");
}
//BIND_END

//BIND_METHOD MatrixFloat dim
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int pos;
  const int *d=obj->getDimPtr();
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, pos, -1);
  if (pos < 1) {
    LUABIND_VECTOR_TO_NEW_TABLE(int, d, obj->getNumDim());
    LUABIND_RETURN_FROM_STACK(-1);
  }
  else LUABIND_RETURN(int, d[pos-1]);
}
//BIND_END

//BIND_METHOD MatrixFloat stride
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int pos;
  const int *s=obj->getStridePtr();
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, pos, -1);
  if (pos < 1) {
    LUABIND_VECTOR_TO_NEW_TABLE(int, s, obj->getNumDim());
    LUABIND_RETURN_FROM_STACK(-1);
  }
  else LUABIND_RETURN(int, s[pos-1]);
}
//BIND_END

//BIND_METHOD MatrixFloat slice
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, table);
  LUABIND_CHECK_PARAMETER(2, table);
  int *coords, *sizes, coords_len, sizes_len;
  bool clone;
  LUABIND_TABLE_GETN(1, coords_len);
  LUABIND_TABLE_GETN(2, sizes_len);
  if (coords_len != sizes_len || coords_len != obj->getNumDim())
    LUABIND_FERROR3("Incorrect number of dimensions, expected %d, "
		    "found %d and %d\n",
		    obj->getNumDim(), coords_len, sizes_len);
  coords = new int[coords_len];
  sizes  = new int[sizes_len];
  LUABIND_TABLE_TO_VECTOR(1, int, coords, coords_len);
  LUABIND_TABLE_TO_VECTOR(2, int, sizes,  sizes_len);
  for (int i=0; i<coords_len; ++i) --coords[i];
  LUABIND_GET_OPTIONAL_PARAMETER(3, bool, clone, false);
  MatrixFloat *obj2 = new MatrixFloat(obj, coords, sizes, clone);
  LUABIND_RETURN(MatrixFloat, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD MatrixFloat select
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  int dim, index;
  LUABIND_GET_PARAMETER(1, int, dim);
  LUABIND_GET_PARAMETER(2, int, index);
  MatrixFloat *obj2 = obj->select(dim-1, index-1);
  LUABIND_RETURN(MatrixFloat, obj2);
}
//BIND_END

//BIND_METHOD MatrixFloat clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int argn;
  argn = lua_gettop(L); // number of arguments
  MatrixFloat *obj2;
  if (argn == 0) obj2 = obj->clone();
  else {
    const char *major;
    LUABIND_GET_OPTIONAL_PARAMETER(1, string, major, "row_major");
    CBLAS_ORDER order=CblasRowMajor;
    if (strcmp(major, "col_major") == 0) order = CblasColMajor;
    else if (strcmp(major, "row_major") != 0)
      LUABIND_FERROR1("Incorrect major order string %s", major);
    obj2 = obj->clone(order);
  }
  LUABIND_RETURN(MatrixFloat,obj2);
}
//BIND_END

//BIND_METHOD MatrixFloat transpose
{
  LUABIND_RETURN(MatrixFloat, obj->transpose());
}
//BIND_END

//BIND_METHOD MatrixFloat isfinite
//DOC_BEGIN
// bool isfinite
/// Devuelve false si algun valor es nan o infinito.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  bool resul=true;
  for (MatrixFloat::iterator it(obj->begin()); resul && it!=obj->end(); ++it)
    //if (!isfinite(obj->data[i])) resul = 0;
    if ((*it) - (*it) != 0.0f) resul = false;
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
  obj->adjustRange(rmin, rmax);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat diag
{
  LUABIND_CHECK_ARGN(==,1);
  float v;
  LUABIND_GET_PARAMETER(1, float, v);
  obj->diag(v);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    LUABIND_FORWARD_CONTAINER_TO_NEW_TABLE(MatrixFloat, float, *obj);
    LUABIND_RETURN_FROM_STACK(-1);
  }
//BIND_END

//BIND_METHOD MatrixFloat min
  {
    int arg_min;
    LUABIND_RETURN(float, obj->min(arg_min));
    LUABIND_RETURN(int, arg_min+1);
  }
//BIND_END

//BIND_METHOD MatrixFloat max
  {
    int arg_max;
    LUABIND_RETURN(float, obj->max(arg_max));
    LUABIND_RETURN(int, arg_max+1);
  }
//BIND_END

//BIND_METHOD MatrixFloat equals
{
  MatrixFloat *other;
  float epsilon;
  LUABIND_GET_PARAMETER(1, MatrixFloat, other);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, epsilon, 1e-04f);
  LUABIND_RETURN(boolean, obj->equals(other, epsilon));
}
//BIND_END

//BIND_METHOD MatrixFloat clamp
  {
    LUABIND_CHECK_ARGN(==, 2);
    float lower,upper;
    LUABIND_GET_PARAMETER(1, float, lower);
    LUABIND_GET_PARAMETER(2, float, upper);
    obj->clamp(lower,upper);
    LUABIND_RETURN(MatrixFloat, obj);
  }
//BIND_END

//BIND_METHOD MatrixFloat add
  {
    int argn;
    argn = lua_gettop(L); // number of arguments
    LUABIND_CHECK_ARGN(==, 1);
    MatrixFloat *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
    if (!obj->sameDim(mat))
      LUABIND_ERROR("matrix add wrong dimensions");
    resul = obj->addition(mat);
    LUABIND_RETURN(MatrixFloat, resul);
  }
//BIND_END

//BIND_METHOD MatrixFloat scalar_add
{
    int argn;
    argn = lua_gettop(L); // number of arguments
    LUABIND_CHECK_ARGN(==, 1);
    float scalar;
    LUABIND_GET_PARAMETER(1, float, scalar);
    obj->scalarAdd(scalar);
    LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat sub
  {
    LUABIND_CHECK_ARGN(==, 1);
    MatrixFloat *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
    if (!obj->sameDim(mat))
      LUABIND_ERROR("matrix sub wrong dimensions");
    resul = obj->substraction(mat);
    LUABIND_RETURN(MatrixFloat, resul);
  }
//BIND_END

//BIND_METHOD MatrixFloat mul
  {
    LUABIND_CHECK_ARGN(==, 1);
    MatrixFloat *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
    resul = obj->multiply(mat);
    if (resul == 0)
      LUABIND_ERROR("matrix mul wrong dimensions");
    LUABIND_RETURN(MatrixFloat, resul);
  }
//BIND_END

//BIND_METHOD MatrixFloat cmul
  {
    LUABIND_CHECK_ARGN(==, 1);
    MatrixFloat *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
    resul = obj->cmul(mat);
    if (resul == 0)
      LUABIND_ERROR("matrix mul wrong dimensions");
    LUABIND_RETURN(MatrixFloat, resul);
  }
//BIND_END

//BIND_METHOD MatrixFloat log
{
  obj->log();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat log1p
{
  obj->log1p();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat exp
{
  obj->exp();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat sqrt
{
  obj->sqrt();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat pow
{
  float value;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, float, value);
  obj->pow(value);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat tanh
{
  obj->tanh();
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat sum
{
  LUABIND_RETURN(float, obj->sum());
}
//BIND_END

//BIND_METHOD MatrixFloat copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  obj->copy(mat);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat axpy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 2);
  float alpha;
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, float, alpha);
  LUABIND_GET_PARAMETER(2, MatrixFloat, mat);
  obj->axpy(alpha, mat);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat gemm
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L,1, "trans_A", "trans_B", "alpha", "A", "B", "beta",
		       (const char *)0);
    bool trans_A, trans_B;
    float alpha;
    float beta;
    MatrixFloat *matA,*matB;
    LUABIND_GET_TABLE_PARAMETER(1, A, MatrixFloat, matA);
    LUABIND_GET_TABLE_PARAMETER(1, B, MatrixFloat, matB);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_B, bool, trans_B, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
    obj->gemm(trans_A ? CblasTrans : CblasNoTrans,
	      trans_B ? CblasTrans : CblasNoTrans,
	      alpha, matA, matB,
	      beta);
    LUABIND_RETURN(MatrixFloat, obj);
  }
//BIND_END

//BIND_METHOD MatrixFloat gemv
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L,1, "trans_A", "alpha", "A", "X", "beta",
		       (const char *)0);
    bool trans_A;
    float alpha;
    float beta;
    MatrixFloat *matA,*matX;
    LUABIND_GET_TABLE_PARAMETER(1, A, MatrixFloat, matA);
    LUABIND_GET_TABLE_PARAMETER(1, X, MatrixFloat, matX);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
    obj->gemv(trans_A ? CblasTrans : CblasNoTrans,
	      alpha, matA, matX,
	      beta);
    LUABIND_RETURN(MatrixFloat, obj);
  }
//BIND_END

//BIND_METHOD MatrixFloat ger
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L,1, "alpha", "X", "Y",
		       (const char *)0);
    float alpha;
    MatrixFloat *matX,*matY;
    LUABIND_GET_TABLE_PARAMETER(1, X, MatrixFloat, matX);
    LUABIND_GET_TABLE_PARAMETER(1, Y, MatrixFloat, matY);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
    obj->ger(alpha, matX, matY);
    LUABIND_RETURN(MatrixFloat, obj);
  }
//BIND_END

//BIND_METHOD MatrixFloat dot
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, MatrixFloat);
    MatrixFloat *matX;
    LUABIND_GET_PARAMETER(1, MatrixFloat, matX);
    LUABIND_RETURN(float, obj->dot(matX));
  }
//BIND_END

//BIND_METHOD MatrixFloat scal
  {
    LUABIND_CHECK_ARGN(==, 1);
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    obj->scal(value);
    LUABIND_RETURN(MatrixFloat, obj);
  }
//BIND_END
 
//BIND_METHOD MatrixFloat norm2
  {
    LUABIND_RETURN(float, obj->norm2());
  }
//BIND_END

//BIND_METHOD MatrixFloat uniform
{
  int lower, upper;
  MTRand *random;
  LUABIND_GET_PARAMETER(1, int, lower);
  LUABIND_GET_PARAMETER(2, int, upper);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
  if (lower < 0)
    LUABIND_ERROR("Allowed only for positive integers");
  if (lower > upper)
    LUABIND_ERROR("First argument must be <= second argument");
  if (random == 0) random = new MTRand();
  IncRef(random);
  if (obj->getMajorOrder() == CblasRowMajor)
    for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it) {
      *it = static_cast<float>(random->randInt(upper - lower) + lower);
    }
  else
    for (MatrixFloat::col_major_iterator it(obj->begin());it!=obj->end();++it) {
      *it = static_cast<float>(random->randInt(upper - lower) + lower);
    }
  DecRef(random);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat uniformf
{
  float lower, upper;
  MTRand *random;
  LUABIND_GET_OPTIONAL_PARAMETER(1, float, lower, 0.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, upper, 1.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
  if (lower > upper)
    LUABIND_ERROR("First argument must be <= second argument");
  if (random == 0) random = new MTRand();
  IncRef(random);
  if (obj->getMajorOrder() == CblasRowMajor)
    for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it)
      *it = random->rand(upper - lower) + lower;
  else
    for (MatrixFloat::col_major_iterator it(obj->begin());it!=obj->end();++it)
      *it = random->rand(upper - lower) + lower;
  DecRef(random);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat linear
{
  int lower, step;
  MTRand *random;
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, lower, 0);
  LUABIND_GET_OPTIONAL_PARAMETER(2, int, step,  1);
  int k=lower;
  for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it, k+=step) {
    *it = static_cast<float>(k);
  }
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat sliding_window
{
  int *sub_matrix_size=0, *offset=0, *step=0, *num_steps=0, *order_step=0;
  int argn = lua_gettop(L); // number of arguments
  const int num_dim = obj->getNumDim();
  if (argn > 1)
    LUABIND_ERROR("incorrect number of arguments");
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1,
		       "offset",
		       "size",
		       "step",
		       "numSteps",
		       "orderStep",
		       (const char*)0);
    
    offset = read_vector(L, "offset", num_dim, 0);
    sub_matrix_size = read_vector(L, "size", num_dim, 0);
    step = read_vector(L, "step", num_dim, 0);
    num_steps = read_vector(L, "numSteps", num_dim, 0);
    order_step = read_vector(L, "orderStep", num_dim, -1);
  }
  SlidingWindow *window = new SlidingWindow(obj,
					    sub_matrix_size,
					    offset,
					    step,
					    num_steps,
					    order_step);
  LUABIND_RETURN(SlidingWindow, window);
  delete[] sub_matrix_size;
  delete[] offset;
  delete[] step;
  delete[] num_steps;
  delete[] order_step;
}
//BIND_END

//BIND_METHOD MatrixFloat is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

//////////////////////////////////////////////////////////////////////

