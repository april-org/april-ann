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
#include "bind_matrix.h"
#include "utilMatrixIO.h"
#include "utilMatrixComplexF.h"
#include "bind_mtrand.h"
#include <cmath> // para isfinite
#include "luabindutil.h"
#include "luabindmacros.h"
#include "bind_math.h"

#define FUNCTION_NAME "read_vector"
static int *read_vector(lua_State *L, const char *key, int num_dim, int add) {
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

int sliding_window_matrix_complex_iterator_function(lua_State *L) {
  SlidingWindowComplexF *obj = lua_toSlidingWindowComplexF(L,1);
  if (obj->isEnd()) {
    lua_pushnil(L);
    return 1;
  }
  // lua_pushSlidingWindow(L, obj);
  MatrixComplexF *mat = obj->getMatrix();
  lua_pushMatrixComplexF(L, mat);
  obj->next();
  return 1;
}

//BIND_END

//BIND_HEADER_H
#include "matrixComplexF.h"
#include "utilLua.h"
#include <cmath> // para isfinite
typedef MatrixComplexF::sliding_window SlidingWindowComplexF;
//BIND_END

//BIND_LUACLASSNAME MatrixComplexF matrixComplex
//BIND_CPP_CLASS MatrixComplexF

//BIND_LUACLASSNAME SlidingWindowComplexF matrix.__sliding_window_complex__
//BIND_CPP_CLASS SlidingWindowComplexF

//BIND_CONSTRUCTOR SlidingWindowComplexF
{
  LUABIND_ERROR("Use matrixComplex.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF get_matrix
{
  MatrixComplexF *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixComplexF, dest, 0);
  LUABIND_RETURN(MatrixComplexF, obj->getMatrix(dest));
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF next
{
  LUABIND_RETURN(SlidingWindowComplexF, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindowComplexF, obj);
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindowComplexF iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_matrix_complex_iterator_function);
  LUABIND_RETURN(SlidingWindowComplexF,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixComplexF
//DOC_BEGIN
// matrixComplex(int dim1, int dim2, ..., table mat=nil)
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
  MatrixComplexF* obj;
  obj = new MatrixComplexF(ndims,dim);
  if (lua_istable(L,argn)) {
    int len;
    LUABIND_TABLE_GETN(argn, len);
    if (len != obj->size())
      LUABIND_FERROR2("Incorrect number of elements at the given table, "
		      "found %d, expected %d", len, obj->size());
    int i=1;
    for (MatrixComplexF::iterator it(obj->begin()); it != obj->end(); ++it,++i) {
      lua_rawgeti(L,argn,i);
      *it = lua_toComplexF(L, -1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixComplexF,obj);
}
//BIND_END

//BIND_CLASS_METHOD MatrixComplexF col_major
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
  MatrixComplexF* obj;
  obj = new MatrixComplexF(ndims,dim,CblasColMajor);
  if (lua_istable(L,argn)) {
    int len;
    LUABIND_TABLE_GETN(argn, len);
    if (len != obj->size())
      LUABIND_FERROR2("Incorrect number of elements at the given table, "
		      "found %d, expected %d", len, obj->size());
    int i=1;
    for (MatrixComplexF::iterator it(obj->begin()); it != obj->end(); ++it,++i) {
      lua_rawgeti(L,argn,i);
      *it = lua_toComplexF(L, -1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixComplexF,obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixComplexF rewrap
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
  MatrixComplexF *new_obj = obj->rewrap(dims, ndims);
  delete[] dims;
  LUABIND_RETURN(MatrixComplexF,new_obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p",
	  (void*)obj,
	  (void*)obj->getRawDataAccess());
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_CLASS_METHOD MatrixComplexF fromFilename
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
  MatrixComplexF *obj;
  if ((obj = readMatrixComplexFFromFile(filename)) == 0)
    LUABIND_ERROR("bad format");
  else LUABIND_RETURN(MatrixComplexF,obj);
}
//BIND_END


//BIND_CLASS_METHOD MatrixComplexF fromString
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
  MatrixComplexF *obj;
  if ((obj = readMatrixComplexFFromString(cs)) == 0)
    LUABIND_ERROR("bad format");
  else LUABIND_RETURN(MatrixComplexF,obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF toFilename
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
  writeMatrixComplexFToFile(obj, filename, is_ascii);
}
//BIND_END

//BIND_METHOD MatrixComplexF toString
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
  int len;
  char *buffer = writeMatrixComplexFToString(obj, is_ascii, len);
  lua_pushlstring(L,buffer,len);
  LUABIND_RETURN_FROM_STACK(-1);
  delete[] buffer;
}
//BIND_END

//BIND_METHOD MatrixComplexF copy_from_table
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
  for (MatrixComplexF::iterator it(obj->begin()); it != obj->end(); ++it,++i) {
    lua_rawgeti(L,1,i);
    *it = lua_toComplexF(L, -1);
  }
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF get
//DOC_BEGIN
// ComplexF get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  ComplexF ret;
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
  LUABIND_RETURN(ComplexF, ret);
}
//BIND_END

//BIND_METHOD MatrixComplexF set
//DOC_BEGIN
// ComplexF set(coordinates,realvalue,imgvalue)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim()+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+2);
  ComplexF f;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,ComplexF,f);
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,ComplexF,f);
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,ComplexF,f);
    (*obj)(coords, obj->getNumDim()) = f;
    delete[] coords;
  }
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixComplexF raw_get
{
  int raw_pos;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  const ComplexF &aux = (*obj)[raw_pos];
  LUABIND_RETURN(ComplexF, aux);
}
//BIND_END

//BIND_METHOD MatrixComplexF raw_set
{
  int raw_pos;
  ComplexF value;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_GET_PARAMETER(2, ComplexF, value);
  (*obj)[raw_pos] = value;
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF fill
//DOC_BEGIN
// void fill(realvalue, imgvalue)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  ComplexF value;
  LUABIND_GET_PARAMETER(1,ComplexF,value);
  obj->fill(value);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF zeros
//DOC_BEGIN
// void zeros(ComplexF value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  obj->zeros();
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF ones
//DOC_BEGIN
// void ones(ComplexF value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  obj->ones();
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF get_use_cuda
{
  LUABIND_RETURN(bool, obj->getCudaFlag());
}
//BIND_END

//BIND_METHOD MatrixComplexF set_use_cuda
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, bool);
  bool v;
  LUABIND_GET_PARAMETER(1,bool, v);
  obj->setUseCuda(v);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF get_major_order
{
  if (obj->getMajorOrder() == CblasRowMajor)
    LUABIND_RETURN(string, "row_major");
  else LUABIND_RETURN(string, "col_major");
}
//BIND_END

//BIND_METHOD MatrixComplexF dim
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

//BIND_METHOD MatrixComplexF stride
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

//BIND_METHOD MatrixComplexF slice
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
  MatrixComplexF *obj2 = new MatrixComplexF(obj, coords, sizes, clone);
  LUABIND_RETURN(MatrixComplexF, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD MatrixComplexF select
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  int dim, index;
  MatrixComplexF *dest;
  LUABIND_GET_PARAMETER(1, int, dim);
  LUABIND_GET_PARAMETER(2, int, index);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixComplexF, dest, 0);
  MatrixComplexF *obj2 = obj->select(dim-1, index-1, dest);
  LUABIND_RETURN(MatrixComplexF, obj2);
}
//BIND_END

//BIND_METHOD MatrixComplexF clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int argn;
  argn = lua_gettop(L); // number of arguments
  MatrixComplexF *obj2;
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
  LUABIND_RETURN(MatrixComplexF,obj2);
}
//BIND_END

//BIND_METHOD MatrixComplexF transpose
{
  LUABIND_RETURN(MatrixComplexF, obj->transpose());
}
//BIND_END

//BIND_METHOD MatrixComplexF isfinite
//DOC_BEGIN
// bool isfinite
/// Devuelve false si algun valor es nan o infinito.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  bool resul=true;
  for (MatrixComplexF::iterator it(obj->begin()); resul && it!=obj->end(); ++it)
    //if (!isfinite(obj->data[i])) resul = 0;
    if (((*it) - (*it)) != ComplexF::zero_zero()) resul = false;
  LUABIND_RETURN(boolean,resul);
}
//BIND_END

//BIND_METHOD MatrixComplexF diag
{
  LUABIND_CHECK_ARGN(==,1);
  ComplexF v;
  LUABIND_GET_PARAMETER(1, ComplexF, v);
  obj->diag(v);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    lua_createtable (L, obj->size()<<1, 0);
    int index = 1;
    for (MatrixComplexF::const_iterator it(obj->begin());
	 it != obj->end();
	 ++it) {
      lua_pushComplexF(L, *it);
      lua_rawseti(L, -2, index++);
    }
    LUABIND_RETURN_FROM_STACK(-1);
  }
//BIND_END

//BIND_METHOD MatrixComplexF equals
{
  MatrixComplexF *other;
  float epsilon;
  LUABIND_GET_PARAMETER(1, MatrixComplexF, other);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, epsilon, 1e-04f);
  LUABIND_RETURN(boolean, obj->equals(other, epsilon));
}
//BIND_END

//BIND_METHOD MatrixComplexF add
  {
    int argn;
    argn = lua_gettop(L); // number of arguments
    LUABIND_CHECK_ARGN(==, 1);
    MatrixComplexF *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, mat);
    if (!obj->sameDim(mat))
      LUABIND_ERROR("matrix add wrong dimensions");
    resul = obj->addition(mat);
    LUABIND_RETURN(MatrixComplexF, resul);
  }
//BIND_END

//BIND_METHOD MatrixComplexF scalar_add
{
    int argn;
    LUABIND_CHECK_ARGN(==, 1);
    ComplexF scalar;
    LUABIND_GET_PARAMETER(1, ComplexF, scalar);
    obj->scalarAdd(scalar);
    LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF sub
  {
    LUABIND_CHECK_ARGN(==, 1);
    MatrixComplexF *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, mat);
    if (!obj->sameDim(mat))
      LUABIND_ERROR("matrix sub wrong dimensions");
    resul = obj->substraction(mat);
    LUABIND_RETURN(MatrixComplexF, resul);
  }
//BIND_END

//BIND_METHOD MatrixComplexF mul
  {
    LUABIND_CHECK_ARGN(==, 1);
    MatrixComplexF *mat,*resul;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, mat);
    resul = obj->multiply(mat);
    if (resul == 0)
      LUABIND_ERROR("matrix mul wrong dimensions");
    LUABIND_RETURN(MatrixComplexF, resul);
  }
//BIND_END

//BIND_METHOD MatrixComplexF sum
{
  ComplexF sum = obj->sum();
  LUABIND_RETURN(float, sum.real());
  LUABIND_RETURN(float, sum.img());
}
//BIND_END

//BIND_METHOD MatrixComplexF copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixComplexF *mat;
  LUABIND_GET_PARAMETER(1, MatrixComplexF, mat);
  obj->copy(mat);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF axpy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 2);
  ComplexF alpha;
  MatrixComplexF *mat;
  LUABIND_GET_PARAMETER(1, ComplexF, alpha);
  LUABIND_GET_PARAMETER(2, MatrixComplexF, mat);
  obj->axpy(alpha, mat);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF gemm
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L,1, "trans_A", "trans_B", "alpha", "A", "B", "beta",
		       (const char *)0);
    bool trans_A, trans_B;
    ComplexF alpha;
    ComplexF beta;
    MatrixComplexF *matA,*matB;
    LUABIND_GET_TABLE_PARAMETER(1, A, MatrixComplexF, matA);
    LUABIND_GET_TABLE_PARAMETER(1, B, MatrixComplexF, matB);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_B, bool, trans_B, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, ComplexF, alpha,
					 ComplexF::one_zero());
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, ComplexF, beta,
					 ComplexF::one_zero());
    obj->gemm(trans_A ? CblasTrans : CblasNoTrans,
	      trans_B ? CblasTrans : CblasNoTrans,
	      alpha, matA, matB,
	      beta);
    LUABIND_RETURN(MatrixComplexF, obj);
  }
//BIND_END

//BIND_METHOD MatrixComplexF gemv
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L,1, "trans_A", "alpha", "A", "X", "beta",
		       (const char *)0);
    bool trans_A;
    ComplexF alpha;
    ComplexF beta;
    MatrixComplexF *matA,*matX;
    LUABIND_GET_TABLE_PARAMETER(1, A, MatrixComplexF, matA);
    LUABIND_GET_TABLE_PARAMETER(1, X, MatrixComplexF, matX);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, ComplexF, alpha, ComplexF::one_zero());
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, ComplexF, beta, ComplexF::one_zero());
    obj->gemv(trans_A ? CblasTrans : CblasNoTrans,
	      alpha, matA, matX,
	      beta);
    LUABIND_RETURN(MatrixComplexF, obj);
  }
//BIND_END

//BIND_METHOD MatrixComplexF ger
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L,1, "alpha", "X", "Y",
		       (const char *)0);
    ComplexF alpha;
    MatrixComplexF *matX,*matY;
    LUABIND_GET_TABLE_PARAMETER(1, X, MatrixComplexF, matX);
    LUABIND_GET_TABLE_PARAMETER(1, Y, MatrixComplexF, matY);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, ComplexF, alpha, ComplexF::one_zero());
    obj->ger(alpha, matX, matY);
    LUABIND_RETURN(MatrixComplexF, obj);
  }
//BIND_END

//BIND_METHOD MatrixComplexF dot
  {
    LUABIND_CHECK_ARGN(==, 1);
    LUABIND_CHECK_PARAMETER(1, MatrixComplexF);
    MatrixComplexF *matX;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, matX);
    LUABIND_RETURN(ComplexF, obj->dot(matX));
  }
//BIND_END

//BIND_METHOD MatrixComplexF scal
  {
    LUABIND_CHECK_ARGN(==, 1);
    ComplexF value;
    LUABIND_GET_PARAMETER(1, ComplexF, value);
    obj->scal(value);
    LUABIND_RETURN(MatrixComplexF, obj);
  }
//BIND_END
 
//BIND_METHOD MatrixComplexF norm2
  {
    LUABIND_RETURN(float, obj->norm2());
  }
//BIND_END

//BIND_METHOD MatrixComplexF uniform
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
    for (MatrixComplexF::iterator it(obj->begin()); it != obj->end(); ++it) {
      *it = ComplexF(static_cast<float>(random->randInt(upper - lower) + lower),
		     0.0f);
    }
  else
    for (MatrixComplexF::col_major_iterator it(obj->begin());it!=obj->end();++it) {
      *it = ComplexF(static_cast<float>(random->randInt(upper - lower) + lower),
		     0.0f);
    }
  DecRef(random);
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF linear
{
  int lower, step;
  MTRand *random;
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, lower, 0);
  LUABIND_GET_OPTIONAL_PARAMETER(2, int, step,  1);
  int k=lower;
  for (MatrixComplexF::iterator it(obj->begin()); it != obj->end(); ++it, k+=step) {
    *it = ComplexF(static_cast<float>(k), 0.0f);
  }
  LUABIND_RETURN(MatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD MatrixComplexF sliding_window
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
  SlidingWindowComplexF *window = new SlidingWindowComplexF(obj,
							    sub_matrix_size,
							    offset,
							    step,
							    num_steps,
							    order_step);
  LUABIND_RETURN(SlidingWindowComplexF, window);
  delete[] sub_matrix_size;
  delete[] offset;
  delete[] step;
  delete[] num_steps;
  delete[] order_step;
}
//BIND_END

//BIND_METHOD MatrixComplexF is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

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

//////////////////////////////////////////////////////////////////////

