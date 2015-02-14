#include "utilLua.h"

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

