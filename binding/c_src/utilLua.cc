#include "utilLua.h"

//devuelve true si consigue leer "name" en la tabla que esté en el top
// y en v[] deja los n primeros valores 
bool leer_int_params(lua_State *L, const char *name, int *v, int n) {
  // table
  lua_pushstring(L, name);  // name (atributo)
  lua_gettable(L,-2);
  if (!lua_istable(L,-1)) {
    lua_pop(L,1);	//remove string
    return false;
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
  if (!lua_istable(L,-1)) {
    lua_pop(L,1);	//remove string
    return false;
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

// toma una tabla en la pila y devuelve la longitud de la misma
// la pila se queda igual, con la tabla en el tope
int table_getn(lua_State *L) {
  lua_pushstring(L,"table");
  lua_gettable(L, LUA_GLOBALSINDEX);
  lua_pushstring(L,"getn");
  lua_gettable(L, -2);
  lua_pushvalue(L,-3);
  lua_call(L,1,1);
  int l = (int)lua_tonumber(L,-1);
  lua_pop(L,2);
  return l;
}

int table_to_char_vector(lua_State *L, char ***vec) {
  int length = table_getn(L);
  char **v = new char*[length];
  for (int i=0; i<length; i++) {
    lua_rawgeti(L, -1, i+1);
    v[i] = (char*)lua_tostring(L, -1);
    lua_pop(L,1);
  }
  *vec = v;
  return length;
}

int table_to_int_vector(lua_State *L, int **vec, int offset) {
  int length = table_getn(L);
  int *v = new int[length];
  for (int i=0; i<length; i++) {
    lua_rawgeti(L, -1, i+1);
    v[i] = (int)lua_tonumber(L, -1) + offset;
    lua_pop(L,1);
  }
  *vec = v;
  return length;
}

int table_to_float_vector(lua_State *L, float **vec) {
  int length = table_getn(L);
  float *v = new float[length];
  for (int i=0; i< length; i++) {
    lua_rawgeti(L, -1, i+1);
    v[i] = lua_tonumber(L, -1);
    lua_pop(L,1);
  }
  *vec = v;
  return length;
}

int table_to_double_vector(lua_State *L, double **vec) {
  int length = table_getn(L);
  double *v = new double[length];
  for (int i=0; i<length; i++) {
    lua_rawgeti(L, -1, i+1);
    v[i] = lua_tonumber(L, -1);
    lua_pop(L,1);
  }
  *vec = v;
  return length;
}

void int_vector_to_table(lua_State *L, int *vec, int sz, int first) {
  lua_newtable(L);
  for (int i=0; i<sz; i++) {
    lua_pushnumber(L,vec[i]);
    lua_rawseti(L,-2,first+i);
  }
}

void float_vector_to_table(lua_State *L, float *vec, int sz, int first) {
  lua_newtable(L);
  for (int i=0; i<sz; i++) {
    lua_pushnumber(L,vec[i]);
    lua_rawseti(L,-2,first+i);
  }
}

void double_vector_to_table(lua_State *L, double *vec, int sz, int first) {
  lua_newtable(L);
  for (int i=0; i<sz; i++) {
    lua_pushnumber(L,vec[i]);
    lua_rawseti(L,-2,first+i);
  }
}

