/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Francisco Zamora-Martinez, Jorge
 * Gorbe-Moya
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
#include <cmath>
#include "bind_function_interface.h"
#include "bind_mtrand.h"
#include "bind_referenced_vector.h"
#include "bind_tokens.h"
#include "error_print.h"
#include "fmeasure.h"
#include "matrix_ext.h"
#include "MersenneTwister.h"

using AprilUtils::constString;
using namespace AprilMath::MatrixExt::Initializers;

//devuelve true si consigue leer "name" en la tabla que esté en el top
// y en v[] deja los n primeros valores 
bool leer_int_params(lua_State *L, const char *name, int *v, int n) {
  // table
  lua_pushstring(L, name);  // name (atributo)
  lua_gettable(L,-2);
  if (lua_isnil(L,-1)) {
    lua_pop(L,1);	//remove string
    return false;
  }
  else if (!lua_istable(L,-1)) {
    ERROR_EXIT1(128, "Expected a table in field %s\n", name);
  }
  // tabla del atributo
  for (int i = 1; i <= n; i++) {
    lua_rawgeti(L, -1, i);
    v[i-1] = (int)luaL_checknumber(L, -1);
    lua_pop(L,1);
  }
  lua_pop(L,1); //tabla del atributo
  
  return true;
}

//idem pero leyendo bool's
bool leer_bool_params(lua_State *L, const char *name, bool *v, int n) {
  // table
  lua_pushstring(L, name);  // name (atributo)
  lua_gettable(L,-2);
  if (lua_isnil(L,-1)) {
    lua_pop(L,1);	//remove string
    return false;
  }
  else if (!lua_istable(L,-1)) {
    ERROR_EXIT1(128, "Expected a table in field %s\n", name);
  }
  // tabla del atributo
  for (int i = 1; i <= n; i++) {
    lua_rawgeti(L, -1, i);
    v[i-1] = (lua_toboolean(L, -1) != 0);
    lua_pop(L,1);
  }
  lua_pop(L,1); //tabla del atributo
  
  return true;                 
}

namespace Basics {

  int dataset_iterator_function(lua_State *L) {
    // se llama con: local var_1, ... , var_n = _f(_s, _var) donde _s es
    // el estado invariante (en este caso el dataset) y _var es var_1 de
    // iteracion anterior (en este caso el indice)
    DataSetFloat *obj = lua_toDataSetFloat(L,1);
    int index = (int)lua_tonumber(L,2) + 1; // le sumamos uno
    if (index > obj->numPatterns()) {
      lua_pushnil(L); return 1;
    }
    lua_pushnumber(L,index);
    int ps = obj->patternSize();
    float *buff = new float[ps];
    obj->getPattern(index-1,buff); // ojito que le RESTAMOS uno
    lua_newtable(L);
    for (int i=0; i < ps; i++) {
      lua_pushnumber(L,buff[i]);
      lua_rawseti(L,-2,i+1);
    }
    delete[] buff;
    return 2;
  }

  int datasetToken_iterator_function(lua_State *L) {
    // se llama con: local var_1, ... , var_n = _f(_s, _var) donde _s es
    // el estado invariante (en este caso el dataset) y _var es var_1 de
    // iteracion anterior (en este caso el indice)
    DataSetToken *obj = lua_toDataSetToken(L,1);
    int index = (int)lua_tonumber(L,2) + 1; // le sumamos uno
    if (index > obj->numPatterns()) {
      lua_pushnil(L); return 1;
    }
    lua_pushnumber(L,index);
    int ps = obj->patternSize();
    AprilUtils::SharedPtr<Token> tk( obj->getPattern(index-1) ); // ojito que le RESTAMOS uno
    lua_pushAuxToken(L,tk);
    return 2;
  }

} // namespace Basics

bool lua_isAuxDataSetToken(lua_State *L, int n) {
  return lua_isDataSetFloat(L,n) || lua_isDataSetToken(L,n);
}

DataSetToken *lua_toAuxDataSetToken(lua_State *L, int n) {
  if (lua_isDataSetFloat(L, n)) {
    DataSetFloat *ds = lua_toDataSetFloat(L,n);
    return new DataSetFloat2TokenWrapper(ds);
  }
  return lua_toDataSetToken(L,n);
}

//BIND_END

//BIND_HEADER_H
#include "datasetFloat.h"
//#include "utilMatrixFloat.h"
#include "matrixFloat.h"
#include "utilLua.h"
#include "bind_matrix.h"
#include "bind_sparse_matrix.h"
#include <cmath> // para sqrt en mean_deviation
#include "bind_mtrand.h"
#include "MersenneTwister.h"
#include "datasetToken.h"

using namespace Basics;
//BIND_END

//BIND_LUACLASSNAME LinearCombConfFloat dataset.linear_comb_conf
//BIND_CPP_CLASS LinearCombConfFloat

//BIND_CONSTRUCTOR LinearCombConfFloat
{
  // la tabla (L,1) contiene un vector de tablas
  // cada una de ellas contiene tuplas (indice,valor)
  LUABIND_CHECK_PARAMETER(1, table);
  int patternsize;
  LUABIND_TABLE_GETN(1, patternsize);
  // vamos a calcular el total de pares (peso,valor)
  int i,j,total = 0,contador;
  for (i=1; i <= patternsize; i++) {
    lua_rawgeti(L,1,i); // tabla i-esima
    int aux;
    LUABIND_TABLE_GETN(2, aux);
    total += aux;
    lua_pop(L,1); // la quitamos de la pila
  }
  // creamos el objeto
  obj = new LinearCombConfFloat(patternsize,total);
  // rellenamos vectores de indices y de pesos
  for (i=1; i <= patternsize; i++) {
    lua_rawgeti(L,1,i); // tabla i-esima
    LUABIND_TABLE_GETN(2, obj->numTuplas[i-1]);
    lua_pop(L,1); // la quitamos de la pila
  }
  contador = 0;
  for (i=1; i <= patternsize; i++) {
    lua_rawgeti(L,1,i); // tabla i-esima
    for (j=1;j <= obj->numTuplas[i-1]; j++) { // recorremos la tabla i-esima
      lua_rawgeti(L,-1,j); // tupla j-esima de tabla i-esima
      lua_rawgeti(L,-1,1); // extraemos el indice
      obj->indices[contador] = (int)lua_tonumber(L,-1);
      lua_pop(L,1); // la quitamos de la pila
      lua_rawgeti(L,-1,2); // extraemos el peso
      obj->pesos[contador] = lua_tonumber(L,-1);
      contador++;
      lua_pop(L,2); // la quitamos de la pila el peso y la tupla j-esima
    }
    lua_pop(L,1); // la quitamos de la pila la tabla i-esima
  }
  // ponemos en obj->numTuplas los valores "acumulados"
  for (i=1; i < patternsize; i++) {
    obj->numTuplas[i] += obj->numTuplas[i-1];
  }

  LUABIND_RETURN(LinearCombConfFloat, obj);
}
//BIND_END


//BIND_LUACLASSNAME DataSetFloat dataset
//BIND_CPP_CLASS DataSetFloat

//BIND_CONSTRUCTOR DataSetFloat
LUABIND_ERROR("use constructor methods: matrix, etc.");
//BIND_END

//BIND_DESTRUCTOR DataSetFloat
{
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat matrix
/* 
   Permite crear un dataset a partir de una matriz.
   Requiere 2 argumentos.
   El primer argumento debe ser la propia matriz.
   El segundo argumento es una tabla con los siguientes campos:

   - offset      -> posicion inicial en la matriz
   - patternSize -> el producto de estos valores da patternSize()
   - stepSize    -> cuanto se avanza en cada dimension
   - numSteps    -> el producto de estos valores da numPatterns()
   - orderStep
   - circular
   - defaultValue

*/

{
  int argn = lua_gettop(L); // number of arguments
  if (argn < 1 || argn > 2)
    LUABIND_ERROR("incorrect number of arguments");
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  if (argn == 2) {
    LUABIND_CHECK_PARAMETER(2, table);
    
    check_table_fields(L, 2,
		       "offset",
		       "patternSize",
		       "stepSize",
		       "numSteps",
		       "orderStep",
		       "circular",
		       "defaultValue",
		       (const char *)0);
  }

  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  MatrixDataSet<float> *obj = new MatrixDataSet<float>(mat);
  //parseamos la tabla con las opciones
  int ndim          = mat->getNumDim();
  int  *int_params  = new int[ndim];
  bool *bool_params = new bool[ndim];

  if (argn > 1) {
    if (leer_int_params(L, "offset", int_params, ndim)) 
      obj->setOffset(int_params);
    if (leer_int_params(L, "patternSize", int_params, ndim)) 
      obj->setSubMatrixSize(int_params);
    if (leer_int_params(L, "stepSize", int_params, ndim)) 
      obj->setStep(int_params);
    if (leer_int_params(L, "numSteps", int_params, ndim)) 
      obj->setNumSteps(int_params);
    if (leer_int_params(L, "orderStep", int_params, ndim)) 
      obj->setOrderStep(int_params);
    if (leer_bool_params(L, "circular", bool_params, ndim)) 
      obj->setCircular(bool_params);
    
    lua_pushstring(L, "defaultValue");
    lua_gettable(L,2);
    if (!lua_isnil(L,-1))
      obj->setDefaultValue((float)luaL_checknumber(L, -1));
  }
  
  delete[] int_params;
  delete[] bool_params;

  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat identity
// Crea un dataset que recorre una matriz identidad
// Es decir, patrones con valor cero excepto un uno en la posición
// determinada por el índice.
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 3);
  int patsize;
  float zerovalue, onevalue;
  LUABIND_GET_PARAMETER(1, int, patsize);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, zerovalue, 0.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(3, float, onevalue,  1.0f);
  DataSetFloat *obj = new IdentityDataSet<float>(patsize,
						 zerovalue,
						 onevalue);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat union
// permite que varios datasets que tienen el mismo patternSize se vean
// como uno solo con un numPatterns igual al sumatorio de los
// numPatterns de los DataSets que lo componen. Requiere un unico
// argumento que es una tabla con los datasets necesarios.
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int num;
  LUABIND_TABLE_GETN(1, num);
  if (num == 0)
    LUABIND_ERROR("at least one dataset is needed");
  DataSetFloat **vds = new DataSetFloat*[num];
  LUABIND_TABLE_TO_VECTOR(1, DataSetFloat, vds, num);
  DataSetFloat *obj = new UnionDataSet<float>(num,vds);
  delete[] vds;
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat join
// Permite juntar las salidas de varios DataSets que tienen el mismo
// numero de patrones (numPatterns). El patternSize del DataSet
// resultante es igual al sumatorio de los patternSize de los DataSets
// que lo componen. Requiere un unico argumento que es una tabla con
// los datasets necesarios.
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int num;
  LUABIND_TABLE_GETN(1, num);
  if (num == 0)
    LUABIND_ERROR("at least one dataset is needed");
  DataSetFloat **vds = new DataSetFloat*[num];
  LUABIND_TABLE_TO_VECTOR(1, DataSetFloat, vds, num);
  DataSetFloat *obj = new JoinDataSet<float>(num,vds);
  delete[] vds;
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat indexed
// el formato indexado permite crear un mapping de indices a
// patterns. Puede ser útil para especificar la salida de una
// clasificación, en tal caso el DataSet sobre el que se basa el
// IndexDataSet representa la salida asociada a cada una de las
// clases.
// Requiere 2 argumentos, y un 3 argumento opcional. El primero es el dataset
// base. El segundo es una tabla con tantos datasets como patternSize tenga el
// dataset base, cada uno de ellos actuará como diccionario. El patternSize del
// dataset index resultante es igual a la suma de los patternSize de cada
// diccionario. El tercer argumento es el primer valor del indice (normalmente 0
// o 1)
{
  LUABIND_CHECK_ARGN(>=, 2);
  LUABIND_CHECK_ARGN(<=, 3);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, table);
  int numdics, firstindex;
  LUABIND_GET_OPTIONAL_PARAMETER(3, int, firstindex, 1);
  
  DataSetFloat *ds = lua_toDataSetFloat(L,1);
  LUABIND_TABLE_GETN(2, numdics);
  if (numdics != ds->patternSize())
    LUABIND_FERROR2("bad number of  dictionary datasets in table (%d instead of %d)",
		    numdics,ds->patternSize());
  DataSetFloat **vds = new DataSetFloat*[numdics+1];
  vds[0] = ds;
  DataSetFloat **vds_aux = vds+1;
  LUABIND_TABLE_TO_VECTOR(2, DataSetFloat, vds_aux, numdics);
  DataSetFloat *obj = new IndexDataSet<float>(vds, firstindex);
  delete[] vds;
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat slice
// permite seleccionar un subconjunto de patterns de un DataSet,
// este subconjunto debe ser un intervalo (para subconjuntos más
// generales se puede utilizar la clase IndexDataSet)
// Requiere 3 argumentos. El primero es el dataset y luego necesita 2
// argumentos que son los indices inicial y final de los patrones que
// se eligen como subconjunto.
{
  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, int);
  LUABIND_CHECK_PARAMETER(3, int);
  DataSetFloat *ds = lua_toDataSetFloat(L,1);
  int ini,fin;
  ini = (int)lua_tonumber(L,2) - 1; // ojito que le RESTAMOS uno
  fin = (int)lua_tonumber(L,3) - 1; // ojito que le RESTAMOS uno
  DataSetFloat *obj = new SubDataSet<float>(ini,fin,ds);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat split
// permite seleccionar un subconjunto de los elementos de salida de
// un DataSet. Es decir, el DatSet resultante tiene el mismo numero
// de patrones (numPatterns) pero un patternSize menor. El
// subconjunto de salida es un intervalo del original
{
  DataSetFloat *ds;
  int ini,fin;
  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, int);
  LUABIND_CHECK_PARAMETER(3, int);
  LUABIND_GET_PARAMETER(1,DataSetFloat,ds);
  LUABIND_GET_PARAMETER(2,int,ini);
  LUABIND_GET_PARAMETER(3,int,fin);
  if (fin < ini)
    LUABIND_ERROR("Impossible interval, initial position is after end position");
  if (fin < 1 || ini < 1)
    LUABIND_ERROR("Initial end end positions start counting from 1");
  if (ini > ds->patternSize() || fin > ds->patternSize())
    LUABIND_ERROR("The interval values are greater than the patternSize()");
  DataSetFloat *obj = new SplitDataSet<float>(ini-1,fin-1,ds);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat linearcomb
// El LinearCombDataSet toma un dataset y realiza una combinación
// lineal de sus patrones para generar el suyo. Si toma un dataset con
// patternSize X y genera sus patrones de patternSize Y necesitaría
// una matriz de XxY, PERO vamos a suponer que casi siempre será una
// matriz muy dispersa y lo que haremos es una lista de pares
// (indice,peso) de modo que cada uno de los Y elementos de salida
// tendrá una lista de esas tuplas.
{
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, LinearCombConfFloat);
  DataSetFloat *ds = lua_toDataSetFloat(L,1);
  LinearCombConfFloat *conf = lua_toLinearCombConfFloat(L,2);
  // creamos el dataset
  DataSetFloat *obj = new LinearCombDataSet<float>(ds,conf);  
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat contextualizer
// permite que un dataset añada contexto de los patrones situados en
// posiciones adyacentes, cosa que más o menos se puede hacer con
// otros datasets, exceptuando lo que ocurre en los bordes. En este
// dataset el patrón se repite en caso necesario para rellenar los
// bordes.
{
  DataSetFloat *ds;
  int izq,der;
  LUABIND_CHECK_ARGN(>=, 3);
  LUABIND_CHECK_ARGN(<=, 4);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, int);
  LUABIND_CHECK_PARAMETER(3, int);
  LUABIND_GET_PARAMETER(1,DataSetFloat,ds);
  LUABIND_GET_PARAMETER(2,int,izq);
  LUABIND_GET_PARAMETER(3,int,der);
  bool reverse=false;
  LUABIND_GET_OPTIONAL_PARAMETER(4,bool,reverse,false);
  DataSetFloat *obj = new ContextualizerDataSet<float>(ds,izq,der,reverse);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat accumulate
{

  // sirve para acumular una serie de valores mediante putpattern se
  // obtiene la suma acumulada mediante getpattern no hay forma de
  // ponerlo a cero excepto al crearlo o haciendo un getpattern y un
  // putpattern del valor negado, salvo errores
  
  int patsz, numpat;
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  LUABIND_GET_PARAMETER(1,int,patsz);
  LUABIND_GET_PARAMETER(2,int,numpat);
  DataSetFloat *obj = new AccumulateDataSet<float>(patsz,numpat);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD DataSetFloat byte
{
  // sirve para representar un conjunto de patrones con una precisión
  // de 8 bits cada valor, puede recibir 2 tipos de parametros:

  // caso 1) recibe un dataset del que hacer una copia, puede recibir
  // los valores min y max o bien calcularlos al vuelo en funcion de los
  // valores maximo y minimo del dataset, en cuyo caso resulta mas
  // costoso al necesitar iterar 2 veces sobre el dataset

  // caso 2) recibe el numero de patrones, el tamanyo de cada patron y
  // los valores min y max, satura si se sale de ese rango. Una vez
  // creado se rellena con putpattern

  int patsz, numpat;
  double min,max,a,b;
  DataSetFloat *obj;
  int argn = lua_gettop(L); // number of arguments
  if (argn >= 1 && lua_isDataSetFloat(L,1)) {
    DataSetFloat *ds = lua_toDataSetFloat(L,1);
    LUABIND_GET_OPTIONAL_PARAMETER(2, double, min, 0.0);
    LUABIND_GET_OPTIONAL_PARAMETER(3, double, max, 1.0);
    numpat = ds->numPatterns();
    patsz  = ds->patternSize();
    float *patt = new float[patsz];
    if (argn == 1) {
      ds->getPattern(0,patt);
      min = max = patt[0];
      for (int i=0; i<numpat;++i) {
	ds->getPattern(i,patt);
	for (int j=0;j<patsz;j++) {
	  if (patt[j] < min) min = patt[j];
	  if (patt[j] > max) max = patt[j];
	}
      }
      if (max - min < 1e-6) max = min + 1e-6;
    }
    b = min;
    a = (max-min)/255;
    obj = new ByteDataSet<float>(patsz,numpat,a,b);
    for (int i=0; i<numpat;++i) {
      ds->getPattern(i,patt);
      obj->putPattern(i,patt);
    }
    delete[] patt;
  } else {
    LUABIND_CHECK_ARGN(>=, 2);
    LUABIND_CHECK_ARGN(<=, 4);
    LUABIND_CHECK_PARAMETER(1, int);
    LUABIND_CHECK_PARAMETER(2, int);
    LUABIND_GET_PARAMETER(1,int,patsz);
    LUABIND_GET_PARAMETER(2,int,numpat);
    LUABIND_GET_OPTIONAL_PARAMETER(3, double, min, 0.0);
    LUABIND_GET_OPTIONAL_PARAMETER(4, double, max, 1.0);
    b	= min;
    a	= (max-min)/255;
    obj = new ByteDataSet<float>(patsz,numpat,a,b);
  }
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//BIND_METHOD DataSetFloat getPattern
//
{
  int index;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_GET_PARAMETER(1,int,index);
  if (index < 1 || index > obj->numPatterns())
    LUABIND_ERROR("index out of range");
  int ps = obj->patternSize();
  float *buff = new float[ps];
  obj->getPattern(index-1,buff); // ojito que le RESTAMOS uno
  LUABIND_VECTOR_TO_NEW_TABLE(float, buff, ps);
  delete[] buff;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD DataSetFloat putPattern
{
  int index;
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_GET_PARAMETER(1,int,index);
  if (index < 1 || index > obj->numPatterns())
    LUABIND_ERROR("index out of range");
  int ps = obj->patternSize();
  float *buff = new float[ps];
  LUABIND_TABLE_TO_VECTOR(2, float, buff, ps);
  obj->putPattern(index-1,buff); // ojito que le RESTAMOS uno
  delete[] buff;
}
//BIND_END

//BIND_METHOD DataSetFloat numPatterns
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(int,obj->numPatterns());
}
//BIND_END

//BIND_METHOD DataSetFloat patternSize
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(int,obj->patternSize());
}
//BIND_END

//BIND_METHOD DataSetFloat toMatrix
// Genera una matriz de dimension 2
{
  LUABIND_CHECK_ARGN(==, 0);
  int dim[2];
  dim[0] = obj->numPatterns();
  dim[1] = obj->patternSize();
  MatrixFloat* mat = new MatrixFloat(2,dim);
  float *d = mat->getRawDataAccess()->getPPALForWrite();
  for (int i=0; i < dim[0]; i++)
    obj->getPattern(i,d+i*dim[1]);
  LUABIND_RETURN(MatrixFloat,mat);
}
//BIND_END

//BIND_METHOD DataSetFloat saveFileAscii
// imprime cada patron en una fila de un fichero
// recibe un filename y opcionalmente la precision
// que es una cadena estilo "%.7f", por defecto "%g"
// ojo que podria petar si es incorrecta!!!!
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  constString cs_formato,cs_filename;
  LUABIND_GET_PARAMETER(1,constString,cs_filename);
  LUABIND_GET_OPTIONAL_PARAMETER(2,constString,cs_formato,constString("%g"));
  int np = obj->numPatterns();
  int ps = obj->patternSize();
  const char *formato  = cs_formato;
  const char *filename = cs_filename;
  FILE *fich = fopen(filename,"w");
  if (fich == 0) {
    LUABIND_FERROR1("Unable to open file \"%s\"",filename);
  } else {
    float *buff = new float[ps];
    if (ps > 0) {
      for (int i=0; i<np; ++i) {
	obj->getPattern(i,buff);
	for (int j=0; j<ps; ++j) {
	  fprintf(fich,formato,buff[j]);
	  if (j<ps-1)
	    fprintf(fich," ");
	  else
	    fprintf(fich,"\n");
	}
      }
    }
    delete[] buff;
    fclose(fich);
  }
}
//BIND_END

//BIND_METHOD DataSetFloat saveFileBinary
// imprime cada patron en una fila de un fichero
// recibe un filename
{
  LUABIND_CHECK_ARGN(==, 1);
  constString cs_filename;
  LUABIND_GET_PARAMETER(1,constString,cs_filename);
  int np = obj->numPatterns();
  int ps = obj->patternSize();
  const char *filename = cs_filename;
  FILE *fich = fopen(filename,"w");
  if (fich == 0) {
    LUABIND_FERROR1("Unable to open file \"%s\"",filename);
  } else {
    float *buff = new float[ps];
    if (ps > 0) {
      for (int i=0; i<np; ++i) {
	obj->getPattern(i,buff);
	fwrite(buff,sizeof(float),ps,fich);
      }
    }
    delete[] buff;
    fclose(fich);
  }
}
//BIND_END

//BIND_METHOD DataSetFloat patterns
// para iterar con un for index,pattern in obj:patterns() do ... end
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,dataset_iterator_function);
  LUABIND_RETURN(DataSetFloat,obj);
  LUABIND_RETURN(int,0);
}
//BIND_END

//BIND_METHOD DataSetFloat mean_deviation
// devuelve 2 tablas, la primera con medias, la segunda con las
// desviaciones típicas
{
  LUABIND_CHECK_ARGN(==, 0);
  int npatt = obj->numPatterns();
  int psize = obj->patternSize();
  float  *patt = new float[psize];
  double *sum  = new double[psize];
  double *sum2 = new double[psize];
  for (int j=0;j<psize;j++) {
    sum[j]  = 0;
    sum2[j] = 0;
  }
  for (int i=0; i < npatt; i++) {
    obj->getPattern(i,patt);
    for (int j=0;j<psize;j++) {
      sum[j]  += patt[j];
      sum2[j] += patt[j]*patt[j];
    }
  }
  lua_newtable(L); // vector medias
  lua_newtable(L); // vector d.tipica
  for (int j=0;j<psize;j++) {
    double Ex  = sum[j] /npatt;
    double Ex2 = sum2[j]/npatt;
    double dtipica = sqrt(Ex2 - Ex*Ex);
    lua_pushnumber(L,Ex); // media
    lua_rawseti(L,-3,j+1);
    lua_pushnumber(L,dtipica);
    lua_rawseti(L,-2,j+1);
  }
  delete[] patt;
  delete[] sum;
  delete[] sum2;
  LUABIND_RETURN_FROM_STACK(1);
  LUABIND_RETURN_FROM_STACK(2);
}
//BIND_END

//BIND_METHOD DataSetFloat normalize_mean_deviation
// recibe 2 tablas, la primera con medias, la segunda con las
// desviaciones típicas
{
  int psize = obj->patternSize();
  int npatt = obj->numPatterns();
  LUABIND_CHECK_ARGN(==, 2);
  if (!lua_istable(L,1) || psize != luaL_len(L,1) ||
      !lua_istable(L,2) || psize != luaL_len(L,2))
    LUABIND_ERROR("dataset normalize_mean_deviation: wrong arguments");

  float  *patt   = new float[psize];
  float  *mean   = new float[psize];
  float  *invdev = new float[psize];

  // cargar media y desviacion tipica de las tablas:
  // LUABIND_TABLE_TO_VECTOR(table, tipo, vector, longitud)
  LUABIND_TABLE_TO_VECTOR(1, float, mean,   psize);
  LUABIND_TABLE_TO_VECTOR(2, float, invdev, psize);
  for (int i=0; i < psize; i++) {
    if (invdev[i] <= 0)
      LUABIND_FERROR2("dataset normalize_mean_deviation method dev[%d] == %f",
		      i,invdev[i]);
    else
      invdev[i] = 1/invdev[i];
  }

  // normalizar:
  for (int i=0; i < npatt; i++) {
    obj->getPattern(i,patt);
    for (int j=0;j<psize;j++) {
      patt[j] = (patt[j]-mean[j])*invdev[j];
    }
    obj->putPattern(i,patt);
  }
  delete[] patt;
  delete[] mean;
  delete[] invdev;
}
//BIND_END

//BIND_METHOD DataSetFloat mean
// devuelve 1 tabla
{
  LUABIND_CHECK_ARGN(==, 0);
  int npatt = obj->numPatterns();
  int psize = obj->patternSize();
  float  *patt = new float[psize];
  double *sum  = new double[psize];
  for (int j=0;j<psize;j++) {
    sum[j]  = 0;
  }
  for (int i=0; i < npatt; i++) {
    obj->getPattern(i,patt);
    for (int j=0;j<psize;j++) {
      sum[j]  += patt[j];
    }
  }
  lua_newtable(L); // vector medias
  for (int j=0;j<psize;j++) {
    double Ex  = sum[j] /npatt;
    lua_pushnumber(L,Ex); // media
    lua_rawseti(L,-2,j+1);
  }
  delete[] patt;
  delete[] sum;
  LUABIND_RETURN_FROM_STACK(1);
}
//BIND_END

//BIND_METHOD DataSetFloat min_max
// devuelve 2 tablas, la primera con los valores minimos, la segunda con
// los valores maximos
{
  LUABIND_CHECK_ARGN(==, 0);
  int npatt = obj->numPatterns();
  int psize = obj->patternSize();
  float  *patt = new float[psize];
  double *vmin = new double[psize];
  double *vmax = new double[psize];
  obj->getPattern(0,patt);
  for (int j=0;j<psize;j++) {
    vmin[j]  = patt[j];
    vmax[j]  = patt[j];
  }
  for (int i=1; i < npatt; i++) {
    obj->getPattern(i,patt);
    for (int j=0;j<psize;j++) {
      if (vmin[j] > patt[j])
	vmin[j] = patt[j];
      if (vmax[j] < patt[j])
	vmax[j] = patt[j];
    }
  }
  // vector de minimos:
  LUABIND_VECTOR_TO_NEW_TABLE(double, vmin, psize);
  // vector de maximos:
  LUABIND_VECTOR_TO_NEW_TABLE(double, vmax, psize);
  delete[] patt;
  delete[] vmin;
  delete[] vmax;
  LUABIND_RETURN_FROM_STACK(1);
  LUABIND_RETURN_FROM_STACK(2);
}
//BIND_END

//BIND_METHOD DataSetFloat cct
// coordinate-cluster transformation (Kurmyshev et al 1996).
// Recibe un threshold, devuelve una matriz de tamaño
// 2**pattersize
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, float);
  float threshold;
  LUABIND_GET_PARAMETER(1,float,threshold);
  int npatt = obj->numPatterns();
  int psize = obj->patternSize();
  if (psize > 12) // fixme: revisar este valor
    LUABIND_ERROR("too big patternsize");
  float  *patt = new float[psize];
  int histogramlength = 1 << psize;
  MatrixFloat* mat = new MatrixFloat(1,&histogramlength);
  matZeros(mat);
  for (int i=1; i < npatt; i++) {
    obj->getPattern(i,patt);
    int index = 0;
    for (int j=0;j<psize;j++) {
      index = (index << 1) | ((patt[j] > threshold) ? 1 : 0);
    }
    (*mat)(index) += 1;
  }
  delete[] patt;
  LUABIND_RETURN(MatrixFloat,mat);
}
//BIND_END

////////////////////////////////////////////////////////////////////////

//BIND_CLASS_METHOD DataSetFloat fmeasure
{
  LUABIND_CHECK_ARGN(==,1);
  DataSetFloat *ds, *GT;
  LUABIND_GET_TABLE_PARAMETER(1, result_dataset, DataSetFloat, ds);
  LUABIND_GET_TABLE_PARAMETER(1, target_dataset, DataSetFloat, GT);
  float PR, RC, f;
  f = Fmeasure(ds, GT, PR, RC);
  LUABIND_RETURN(float,  f);
  LUABIND_RETURN(float, PR);
  LUABIND_RETURN(float, RC);
}
//BIND_END

////////////////////////////////////////////////////////////////////////

//BIND_CLASS_METHOD DataSetFloat bit
{
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "numPatterns", "patternSize", "dataset",
		     (const char *)0);
  DataSetFloat	*ds;
  int		 nump, patsize;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, numPatterns, int, nump,    -1);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, patternSize, int, patsize, -1);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, dataset,     DataSetFloat, ds, 0);
  if (ds != 0 && (nump != -1 || patsize != -1))
    LUABIND_ERROR("dataset.bit: dataset forbide numPatterns "
		  "and patternSize in arguments table!!!\n");
  if ((nump == -1 || patsize == -1) && ds == 0)
    LUABIND_ERROR("dataset.bit: needed numPatterns"
		  "and patternSize in arguments table!!!\n");
  if (ds) {
    nump    = ds->numPatterns();
    patsize = ds->patternSize();
  }
  DataSetFloat	*obj = new BitDataSet<float>(nump, patsize);
  if (ds) {
    float *pat = new float[patsize];
    for (int i=0; i<nump; ++i) {
      ds->getPattern(i, pat);
      obj->putPattern(i, pat);
    }
  }
  LUABIND_RETURN(DataSetFloat, obj);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////

//BIND_CLASS_METHOD DataSetFloat sparse
/* 
   Permite crear un dataset a partir de una matriz sparse, pero los patrones
   seran densos.
*/

{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, SparseMatrixFloat);
  SparseMatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, SparseMatrixFloat, mat);
  SparseMatrixDatasetFloat *obj = new SparseMatrixDatasetFloat(mat);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////

//BIND_CLASS_METHOD DataSetFloat short_list
{
  DataSetFloat *ds;
  check_table_fields(L, 1, "ds", "unk_word", "short_list_size",
		     (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, ds, DataSetFloat, ds);
  int unk_word, short_list_size;
  LUABIND_GET_TABLE_PARAMETER(1, unk_word, int, unk_word);
  LUABIND_GET_TABLE_PARAMETER(1, short_list_size, int, short_list_size);
  DataSetFloat *obj = new ShortListDataSetFloat(ds, short_list_size, unk_word);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME IndexFilterDataSetFloat dataset.index_filter
//BIND_CPP_CLASS    IndexFilterDataSetFloat
//BIND_SUBCLASS_OF  IndexFilterDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR IndexFilterDataSetFloat
{
  DataSetFloat *ds;
  ReferencedVectorUint *vec;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  LUABIND_GET_PARAMETER(2, ReferencedVectorUint, vec);
  DataSetFloat *obj = new IndexFilterDataSetFloat(ds, vec);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END


//////////////////////////////////////////

//BIND_LUACLASSNAME PerturbationDataSetFloat dataset.perturbation
//BIND_CPP_CLASS    PerturbationDataSetFloat
//BIND_SUBCLASS_OF  PerturbationDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR PerturbationDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "dataset", "mean", "variance",  "random",
		     (const char *)0);
  DataSetFloat *ds;
  MTRand *random;
  double mean, variance;
  LUABIND_GET_TABLE_PARAMETER(1, dataset, DataSetFloat, ds    );
  LUABIND_GET_TABLE_PARAMETER(1, random,  MTRand,       random);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, mean,     double, mean,     0.0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, variance, double, variance, 1.0);
  DataSetFloat *obj = new PerturbationDataSetFloat(ds, random, mean, variance);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME SaltNoiseDataSetFloat dataset.salt_noise
//BIND_CPP_CLASS    SaltNoiseDataSetFloat
//BIND_SUBCLASS_OF  SaltNoiseDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR SaltNoiseDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "dataset", "vd", "zero",  "random",
		     (const char *)0);
  DataSetFloat *ds;
  MTRand *random;
  double vd;
  float zero;
  LUABIND_GET_TABLE_PARAMETER(1, dataset, DataSetFloat, ds    );
  LUABIND_GET_TABLE_PARAMETER(1, random,  MTRand,       random);
  LUABIND_GET_TABLE_PARAMETER(1, vd,      double,       vd);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, zero, float, zero, 0.0f);
  DataSetFloat *obj = new SaltNoiseDataSetFloat(ds, random, vd, zero);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME SaltPepperNoiseDataSetFloat dataset.salt_pepper_noise
//BIND_CPP_CLASS    SaltPepperNoiseDataSetFloat
//BIND_SUBCLASS_OF  SaltPepperNoiseDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR SaltPepperNoiseDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "dataset", "zero", "one", "random", "prob",
                     (const char *)0);
  DataSetFloat *ds;
  MTRand *random;
  double prob;
  float zero, one;
  LUABIND_GET_TABLE_PARAMETER(1, dataset, DataSetFloat, ds    );
  LUABIND_GET_TABLE_PARAMETER(1, random,  MTRand,       random);
  LUABIND_GET_TABLE_PARAMETER(1, prob,    double,       prob);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, zero, float, zero, 0.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, one,  float, one,  1.0f);
  DataSetFloat *obj = new SaltPepperNoiseDataSetFloat(ds, random,
                                                      prob, zero, one);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME StepDataSetFloat dataset.step
//BIND_CPP_CLASS    StepDataSetFloat
//BIND_SUBCLASS_OF  StepDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR StepDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, int);

  DataSetFloat *ds;
  int step;

  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  LUABIND_GET_PARAMETER(2, int, step);
  DataSetFloat *obj = new StepDataSetFloat(ds, step);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME DerivDataSetFloat dataset.deriv
//BIND_CPP_CLASS    DerivDataSetFloat
//BIND_SUBCLASS_OF  DerivDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR DerivDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "dataset", "deriv0", "deriv1",  "deriv2",
		     (const char *)0);
  DataSetFloat *ds;
  bool deriv0, deriv1, deriv2;
  LUABIND_GET_TABLE_PARAMETER(1, dataset, DataSetFloat, ds);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, deriv0, bool, deriv0, true);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, deriv1, bool, deriv1, true);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, deriv2, bool, deriv2, true);
  DataSetFloat *obj = new DerivDataSetFloat(ds, deriv0, deriv1, deriv2);
  LUABIND_RETURN(DataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME CacheDataSetFloat dataset.cache
//BIND_CPP_CLASS    CacheDataSetFloat
//BIND_SUBCLASS_OF  CacheDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR CacheDataSetFloat
{
  DataSetFloat *ds;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  int cache_size;
  LUABIND_GET_PARAMETER(4, int, cache_size);
  int voc_size;
  LUABIND_TABLE_GETN(2, voc_size);
  int  **word2index = new int*[voc_size+1];
  int   *word2index_sizes = new int[voc_size+1];
  float *decays = new float[cache_size+1];
  LUABIND_TABLE_TO_VECTOR(3, float, (decays+1), cache_size);
  float near_zero;
  LUABIND_GET_PARAMETER(5, float, near_zero);
  int begin_token_id, end_token_id, null_token_id, cache_stop_token_id;
  LUABIND_GET_PARAMETER(6, int, begin_token_id);
  LUABIND_GET_PARAMETER(7, int, end_token_id);
  LUABIND_GET_PARAMETER(8, int, null_token_id);
  LUABIND_GET_PARAMETER(9, int, cache_stop_token_id);
  
  // el 0 esta reservado para la palabra vacia
  word2index[0] = 0;
  word2index_sizes[0] = 0;
  decays[0] = 0.0f;
  for (int i=1; i<=voc_size; ++i) {
    int sz;
    lua_rawgeti(L, 2, i);
    LUABIND_TABLE_GETN(-1, sz);
    word2index_sizes[i] = sz;
    if (sz == 0) word2index[i] = 0;
    else {
      word2index[i] = new int[sz];
      LUABIND_TABLE_TO_VECTOR(-1, int, word2index[i], sz);
    }
    lua_pop(L, 1);
  }

  obj = new CacheDataSetFloat(ds,
			      word2index,
			      word2index_sizes,
			      decays,
			      voc_size,
			      cache_size,
			      near_zero,
			      begin_token_id,
			      end_token_id,
			      null_token_id,
			      cache_stop_token_id);
  LUABIND_RETURN(DataSetFloat, obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME SubAndDivNormalizationDataSetFloat dataset.sub_and_div_normalization
//BIND_CPP_CLASS    SubAndDivNormalizationDataSetFloat
//BIND_SUBCLASS_OF  SubAndDivNormalizationDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR SubAndDivNormalizationDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, table);
  LUABIND_CHECK_PARAMETER(3, table);
  DataSetFloat *ds;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  int n,m;
  LUABIND_TABLE_GETN(2,n);
  LUABIND_TABLE_GETN(3,m);
  if (n != ds->patternSize() || m != ds->patternSize())
    LUABIND_ERROR("Incorrect size in table arguments\n");
  float *sub = new float[n];
  float *div = new float[n];
  LUABIND_TABLE_TO_VECTOR(2, float, sub, n);
  LUABIND_TABLE_TO_VECTOR(3, float, div, n);
  obj = new SubAndDivNormalizationDataSet<float>(ds,sub,div);
  delete[] sub;
  delete[] div;
  LUABIND_RETURN(SubAndDivNormalizationDataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME ClampDataSetFloat dataset.clamp
//BIND_CPP_CLASS    ClampDataSetFloat
//BIND_SUBCLASS_OF  ClampDataSetFloat DataSetFloat

//BIND_CONSTRUCTOR ClampDataSetFloat
{
  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  LUABIND_CHECK_PARAMETER(2, float);
  LUABIND_CHECK_PARAMETER(3, float);
  DataSetFloat *ds;
  float lower, upper;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  LUABIND_GET_PARAMETER(2, float, lower);
  LUABIND_GET_PARAMETER(3, float, upper);
  obj = new ClampDataSetFloat(ds,lower,upper);
  LUABIND_RETURN(ClampDataSetFloat,obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME DataSetToken dataset.token
//BIND_CPP_CLASS    DataSetToken

//BIND_CONSTRUCTOR DataSetToken
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD DataSetToken numPatterns
{
  LUABIND_RETURN(int, obj->numPatterns());
}
//BIND_END

//BIND_METHOD DataSetToken patternSize
{
  LUABIND_RETURN(int, obj->patternSize());
}
//BIND_END

//BIND_METHOD DataSetToken getPattern
{
  int index;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_GET_PARAMETER(1,int,index);
  if (index < 1 || index > obj->numPatterns())
    LUABIND_ERROR("index out of range");
  AprilUtils::SharedPtr<Token> token( obj->getPattern(index-1) ); // ojito que le RESTAMOS uno
  LUABIND_RETURN(AuxToken, token);
}
//BIND_END

//BIND_METHOD DataSetToken getPatternBunch
{
  unsigned int bunch_size;
  int *indexes;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  LUABIND_TABLE_GETN(1,bunch_size);
  indexes = new int[bunch_size];
  LUABIND_TABLE_TO_VECTOR_SUB1(1,uint,indexes,bunch_size);
  AprilUtils::SharedPtr<Token> token( obj->getPatternBunch(indexes,bunch_size) );
  delete[] indexes;
  LUABIND_RETURN(AuxToken, token);
}
//BIND_END

//BIND_METHOD DataSetToken putPattern
{
  int index;
  AprilUtils::SharedPtr<Token> pattern;
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, AuxToken);
  LUABIND_GET_PARAMETER(1,int,index);
  LUABIND_GET_PARAMETER(2,AuxToken,pattern);
  if (index < 1 || index > obj->numPatterns())
    LUABIND_ERROR("index out of range");
  obj->putPattern(index-1, pattern.get()); // ojito que le RESTAMOS uno
}
//BIND_END

//BIND_METHOD DataSetToken putPatternBunch
{
  unsigned int bunch_size;
  int *indexes;
  AprilUtils::SharedPtr<Token> pattern;
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, table);
  LUABIND_CHECK_PARAMETER(2, AuxToken);
  LUABIND_GET_PARAMETER(2,AuxToken,pattern);
  LUABIND_TABLE_GETN(1,bunch_size);
  indexes = new int[bunch_size];
  LUABIND_TABLE_TO_VECTOR_SUB1(1,uint,indexes,bunch_size);
  obj->putPatternBunch(indexes,bunch_size,pattern.get());
  delete[] indexes;
}
//BIND_END

//BIND_METHOD DataSetToken patterns
// para iterar con un for index,pattern in obj:patterns() do ... end
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,datasetToken_iterator_function);
  LUABIND_RETURN(DataSetToken,obj);
  LUABIND_RETURN(int,0);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME UnionDataSetToken dataset.token.union
//BIND_CPP_CLASS    UnionDataSetToken
//BIND_SUBCLASS_OF  UnionDataSetToken DataSetToken

//BIND_CONSTRUCTOR UnionDataSetToken
{
  LUABIND_CHECK_ARGN(<=,1);
  int argn = lua_gettop(L);
  if (argn == 1) {
    unsigned int size;
    LUABIND_CHECK_PARAMETER(1, table);
    LUABIND_TABLE_GETN(1, size);
    if (size < 2)
      LUABIND_ERROR("UnionDataSetToken needs a Lua table with two or "
		    "more DataSetToken\n");
    DataSetToken **ds_array = new DataSetToken*[size];
    LUABIND_TABLE_TO_VECTOR(1, AuxDataSetToken, ds_array, size);
    obj = new UnionDataSetToken(ds_array, size);
    delete[] ds_array;
  }
  else obj = new UnionDataSetToken();
  LUABIND_RETURN(UnionDataSetToken, obj);
}
//BIND_END

//BIND_METHOD UnionDataSetToken push_back
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, AuxDataSetToken);
  DataSetToken *ds;
  LUABIND_GET_PARAMETER(1, AuxDataSetToken, ds);
  obj->push_back(ds);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME DataSetTokenVector dataset.token.vector
//BIND_CPP_CLASS    DataSetTokenVector
//BIND_SUBCLASS_OF  DataSetTokenVector DataSetToken

//BIND_CONSTRUCTOR DataSetTokenVector
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, int);
  int pattern_size;
  LUABIND_GET_PARAMETER(1, int, pattern_size);
  obj = new DataSetTokenVector(pattern_size);
  LUABIND_RETURN(DataSetTokenVector, obj);
}
//BIND_END

//BIND_METHOD DataSetTokenVector push_back
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, AuxToken);
  AprilUtils::SharedPtr<Token> token;
  LUABIND_GET_PARAMETER(1, AuxToken, token);
  obj->push_back(token.get());
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME DataSetFloat2TokenWrapper dataset.token.wrapper
//BIND_CPP_CLASS    DataSetFloat2TokenWrapper
//BIND_SUBCLASS_OF  DataSetFloat2TokenWrapper DataSetToken

//BIND_CONSTRUCTOR DataSetFloat2TokenWrapper
{
  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_CHECK_ARGN(<=,2);
  int argn = lua_gettop(L); // number of arguments
  LUABIND_CHECK_PARAMETER(1, DataSetFloat);
  DataSetFloat *ds;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  obj = new DataSetFloat2TokenWrapper(ds);
  LUABIND_RETURN(DataSetFloat2TokenWrapper, obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME DataSetTokenFilter dataset.token.filter
//BIND_CPP_CLASS    DataSetTokenFilter
//BIND_SUBCLASS_OF  DataSetTokenFilter DataSetToken

//BIND_CONSTRUCTOR DataSetTokenFilter
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, AuxDataSetToken);
  LUABIND_CHECK_PARAMETER(2, FunctionInterface);
  DataSetToken *ds;
  FunctionInterface *filter;
  LUABIND_GET_PARAMETER(1, AuxDataSetToken, ds);
  LUABIND_GET_PARAMETER(2, FunctionInterface, filter);
  obj = new DataSetTokenFilter(ds,filter);
  LUABIND_RETURN(DataSetTokenFilter, obj);
}
//BIND_END

//////////////////////////////////////////

//BIND_LUACLASSNAME SparseMatrixDataSetToken dataset.token.sparse_matrix
//BIND_CPP_CLASS    SparseMatrixDataSetToken
//BIND_SUBCLASS_OF  SparseMatrixDataSetToken DataSetToken

//BIND_CONSTRUCTOR SparseMatrixDataSetToken
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, SparseMatrixFloat);
  SparseMatrixFloat *sparse_matrix;
  LUABIND_GET_PARAMETER(1, SparseMatrixFloat, sparse_matrix);
  obj = new SparseMatrixDataSetToken(sparse_matrix);
  LUABIND_RETURN(SparseMatrixDataSetToken, obj);
}
//BIND_END

