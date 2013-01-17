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
//BIND_END

//BIND_HEADER_H
#include "MersenneTwister.h"
#include "dice.h"
#include "utilLua.h"
//BIND_END

//BIND_LUACLASSNAME MTRand random
//BIND_CPP_CLASS MTRand

//BIND_CONSTRUCTOR MTRand
//DOC_BEGIN
// random()
//DOC_END
//DOC_BEGIN
// random(uint32_t n)
/// @param n Valor de la semilla
//DOC_END
//DOC_BEGIN
// random(uint32_t n[])
/// @param n[] Vector con semillas para inicializar
//DOC_END
{
  int argn = lua_gettop(L); /* number of arguments */
  if (argn == 0)
    obj = new MTRand(); // auto-initialize with /dev/urandom or time()
  // and clock()
  else if (argn == 1)
    // initialize with a simple uint32
    obj = new MTRand((uint32_t)luaL_checknumber(L, 1));
  else {
    uint32_t *v = new uint32_t[argn];
    int i;
    for (i=1; i <= argn; i++) {
      if (!lua_isnumber(L,i)) {
	delete[] v;
	lua_pushstring(L,"incorrect argument for mtrand initialization");
	lua_error(L);
      }
      v[i-1] = (uint32_t)lua_tonumber(L,i);
    }
    obj = new MTRand(v, argn);
    delete[] v;
  }
  LUABIND_RETURN(MTRand, obj);
}
//BIND_END

//BIND_DESTRUCTOR MTRand
{
}
//BIND_END

//BIND_METHOD MTRand rand
//DOC_BEGIN
// double rand()
/** Genera un valor aleatorio del intervalo [0,n], donde
 * n es el argumento recibido. Si no se le da ningún valor se
 * considera n=1.
 */
//DOC_END
//DOC_BEGIN
// double rand(double n)
/// Genera un valor aleatorio del intervalo [0,n], donde
/// n es el argumento recibido. Si no se le da ningún valor se
/// considera n=1.
/// @param n Valor del intervalo [0,n] para generar el número aleatorio
//DOC_END
{
  LUABIND_CHECK_ARGN(<=, 1);
  double n;
  LUABIND_GET_OPTIONAL_PARAMETER(1,double,n,1);
  LUABIND_RETURN(number,obj->rand(n));
}
//BIND_END

//BIND_METHOD MTRand randExc
//DOC_BEGIN
// double randExc(double n = 1.0)
/// Genera un valor aleatorio del intervalo [0,n), donde
/// n es el argumento recibido. Si no se le da ningún valor se
/// considera n=1.
/// @param n Valor del intervalo [0,n) para generar el número aleatorio
//DOC_END
{
  LUABIND_CHECK_ARGN(<=, 1);
  double n;
  LUABIND_GET_OPTIONAL_PARAMETER(1,double,n,1);
  LUABIND_RETURN(number,obj->randExc(n));
}
//BIND_END

//BIND_METHOD MTRand randDblExc
//DOC_BEGIN
// double randDblExc(double n = 1.0)
/// Genera un valor aleatorio del intervalo (0,n), donde
/// n es el argumento recibido. Si no se le da ningún valor se
/// considera n=1.
/// @param n Valor del intervalo (0,n) para generar el número aleatorio
//DOC_END
{
  LUABIND_CHECK_ARGN(<=, 1);
  double n;
  LUABIND_GET_OPTIONAL_PARAMETER(1,double,n,1);
  LUABIND_RETURN(number,obj->randDblExc(n));
}
//BIND_END

//BIND_METHOD MTRand randInt
// integer in [0,2^32-1]
// integer in [0,x] if integer x is given
// integer in [x,y] if integers x and y are given
//DOC_BEGIN
// uint32_t randInt(uint32_t ini = 0, uint32_t fin = 2^32 - 1)
/// Genera un valor aleatorio del intervalo [ini, fin], donde
/// x e y son los argumentos recibidos. Por defecto ini=0 y fin = 2^32 - 1.
/// @param ini Punto inicial del intervalo
/// @param fin Punto final del intervalo
//DOC_END
{
  LUABIND_CHECK_ARGN(<=, 2);
  int x,y,resul; // FIXME las hacemos uint32 ???
  int argn = lua_gettop(L);  /* number of arguments */
  if (argn == 2) {
    LUABIND_GET_PARAMETER(1,int,x); // casting int a uint32
    LUABIND_GET_PARAMETER(2,int,y); // casting int a uint32
    if (y < x)
      LUABIND_ERROR("first argument must be <= second argument");
    resul = x+obj->randInt(y-x);
  } else if (argn == 1) {
    LUABIND_GET_PARAMETER(1,int,x); // casting int a uint32
    resul = obj->randInt(x);
  } else
    resul = obj->randInt();
  LUABIND_RETURN(number,resul);
}
//BIND_END

//BIND_METHOD MTRand shuffle
//DOC_BEGIN
// int[] shuffle(int size)
/// @param size Tamaño de la secuencia que queremos desordenar
//DOC_END
//DOC_BEGIN
// vector shuffle(vector)
/// @param vector Vector a desordenar, devuelve OTRO vector desordenado
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  int size = 0; // inicializamos para evitar warning
  int *vector = 0;
  bool is_number = true;
  if (lua_isnumber(L,1)) { 
    // caso en que recibimos la talla de la tabla a generar
    LUABIND_GET_PARAMETER(1, int, size);
    if (size <= 0)
      LUABIND_ERROR("random shuffle: size must be >= 0");
  } else if (lua_istable(L,1)) {
    // recibimos una tabla tipo vector y devolvemos otra con los
    // mismos elementos pero permutados
    size = lua_objlen(L,1);
    is_number = false;
  } else LUABIND_ERROR("random shuffle: an int or a table expected");
  vector = new int[size];
  obj->shuffle(size,vector);
  // lua_createtable(L,narr,nrec) where narr array,  nrec non-array
  lua_createtable(L, size, 0);
  //
  if (is_number) {
    // FIXME: USAR MACRO PARA CARGAR VECTOR???
    for (int i=0; i < size; i++) {
      lua_pushnumber(L,vector[i]+1);
      lua_rawseti(L,2,i+1);
    }
  } else {
    for (int i=0; i < size; i++) {
      lua_rawgeti(L,1,vector[i]+1);
      lua_rawseti(L,2,i+1);
    }
  }
  delete[] vector;
  return 1;
}
//BIND_END

//BIND_METHOD MTRand choose
//DOC_BEGIN
// element choose(vector)
/// @param vector
/// elige un elemento al azar del vector
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  if (!lua_istable(L,1)) {
    LUABIND_ERROR("random choose: a table expected");
  }
  int len   = lua_objlen(L,-1);
  if (len > 0) {
    int which = 1+obj->randInt(len-1);
    lua_rawgeti(L,-1,which);
    LUABIND_RETURN_FROM_STACK(-1);
  }
}
//BIND_END

//BIND_METHOD MTRand randNorm
//DOC_BEGIN
// double randNorm(double mean, double variance)
/// @param mean Media de la función gaussiana
/// @param variance Varianza de la función gaussiana
//DOC_END
{
  double mean, variance;
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_GET_PARAMETER(1, double, mean);
  LUABIND_GET_PARAMETER(2, double, variance);
  LUABIND_RETURN(number,obj->randNorm(mean,variance));
}
//BIND_END

//BIND_METHOD MTRand seed
//DOC_BEGIN
// void seed()
//DOC_END
//DOC_BEGIN
// void seed(uint32_t n)
/// @param n Valor de la semilla inicial
//DOC_END
//DOC_BEGIN
// void seed(uint32_t v[])
/// @param v[] Vector de semillas para inicializar el objeto
//DOC_END
{
  int argn = lua_gettop(L);  /* number of arguments */
  if (argn == 0) {
    obj->seed();
  } else if (argn == 1) {
    obj->seed((uint32_t)luaL_checknumber(L, 1));
  } else if (argn > 1) {
    uint32_t *v = new uint32_t[argn];
    int i;
    for (i=1; i <= argn; i++) {
      v[i-1] = (uint32_t)luaL_checknumber(L, i);
    }
    obj->seed(v,argn);
    delete[] v;
  }
}
//BIND_END

//BIND_METHOD MTRand clone
//DOC_BEGIN
// random *clone()
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  MTRand *theclone = new MTRand(*obj);
  LUABIND_RETURN(MTRand,theclone);
}
//BIND_END

//BIND_METHOD MTRand toTable
//DOC_BEGIN
// table toTable()
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  uint32_t randState[ MTRand::SAVE ];
  obj->save( randState );
  lua_newtable(L);
  for (unsigned int i=0; i < MTRand::SAVE; i++) {
    lua_pushnumber(L,randState[i]);
    lua_rawseti(L,-2,i+1);
  }
  return 1;
}
//BIND_END

//BIND_METHOD MTRand fromTable
//DOC_BEGIN
// void fromTable(table t)
/// @param 
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  uint32_t randState[ MTRand::SAVE ];
  for (int i=1; i <= (int)MTRand::SAVE; i++) {
    lua_rawgeti(L,1,i);
    randState[i-1] = (uint32_t)luaL_checknumber(L, -1);
    lua_remove(L,-1);
  }
  obj->load( randState );	
}
//BIND_END

//BIND_LUACLASSNAME dice random.dice
//BIND_CPP_CLASS dice

//BIND_CONSTRUCTOR dice
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  // leer vector de doubles:
  double *v;
  int length = table_to_double_vector(L, &v);
  for (int i=0; i<length; i++) {
    if (v[i] <= 0.0) {
      LUABIND_ERROR("random.dice values should be positive");
      // FIXME cambiar cuando exista version de error con formato
      //LUABIND_ERROR("random.dice at index %d value should be positive",i+1);
    }
  }
  obj = new dice(length,v);
  delete[] v;
  LUABIND_RETURN(dice,obj);
}
//BIND_END

//BIND_DESTRUCTOR dice
{
}
//BIND_END

//BIND_METHOD dice outcomes
//DOC_BEGIN
// int outcomes()
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(number,obj->get_outcomes());
}
//BIND_END

//BIND_METHOD dice thrown
//DOC_BEGIN
// int thrown(random *generator)
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, MTRand);
  MTRand *generator = lua_toMTRand(L,1);
  LUABIND_RETURN(number,obj->thrown(generator)+1);
}
//BIND_END

