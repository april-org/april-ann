#ifndef UTILLUA_H
#define UTILLUA_H

extern "C"{
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

//devuelve true si consigue leer "name" en la tabla que esté en el top
// y en v[] deja los n primeros valores 
bool leer_int_params(lua_State *L, const char *name, int *v, int n);

//idem pero leyendo bool's
bool leer_bool_params(lua_State *L, const char *name, bool *v, int n);

// toma una tabla en la pila y devuelve la longitud de la misma
// la pila se queda igual, con la tabla en el tope
#define table_getn(L) luaL_len((L),-1);

// cargan una tabla que contiene valores de tipo int, float o double
// devuelve el tamaño del vector reservado usando new tipo[...]  que
// se devuelve por el argumento. Es responsabilidad del que llama
// liberar esa memoria usando delete[]
int table_to_int_vector(lua_State *L, int **vec, int offset=0);
int table_to_float_vector(lua_State *L, float **vec);
int table_to_double_vector(lua_State *L, double **vec);
int table_to_char_vector(lua_State *L, char ***vec);

// crea una tabla lua en la que pone un vector vec de enteros de talla
// sz
void int_vector_to_table(   lua_State *L, int    *vec, int sz, int first=1);
void float_vector_to_table( lua_State *L, float  *vec, int sz, int first=1);
void double_vector_to_table(lua_State *L, double *vec, int sz, int first=1);

#endif // UTILLUA_H

