#ifndef LUABINDUTIL_H
#define LUABINDUTIL_H

#include <stdint.h>

extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

int equaluserdata(lua_State *L);

typedef char* NEW_STRING;


#define lua_strlen(L,idx) luaL_len((L),(idx))

bool   lua_tobool(lua_State *L, int idx);
int    lua_toint(lua_State *L, int idx);
int64_t lua_toint64(lua_State *L, int idx);
unsigned int lua_touint(lua_State *L, int idx);
float  lua_tofloat(lua_State *L, int idx);
double lua_todouble(lua_State *L, int idx);
char  *lua_toNEW_STRING(lua_State *L, int idx);
char   lua_tochar(lua_State *L, int idx);
void  *lua_tolightuserdata(lua_State *L, int idx);
#ifdef CONSTSTRING_H
inline AprilUtils::constString lua_toconstString(lua_State *L, int idx) {
  return AprilUtils::constString(lua_tostring(L, idx),lua_strlen(L, idx));
}
#endif

bool   lua_isFILE(lua_State *L, int idx);
FILE*  lua_toFILE(lua_State *L, int idx);

bool   lua_isbool(lua_State *L, int idx);
bool   lua_isint(lua_State *L, int idx);
bool   lua_isint64(lua_State *L, int idx);
bool   lua_isuint(lua_State *L, int idx);
bool   lua_isfloat(lua_State *L, int idx);
bool   lua_isdouble(lua_State *L, int idx);
bool   lua_ischar(lua_State *L, int idx);
bool   lua_isNEW_STRING(lua_State *L, int idx);
bool   lua_isconstString(lua_State *L, int idx);

void   lua_pushbool(lua_State *L, bool value);
void   lua_pushint(lua_State *L, int value);
void   lua_pushint64(lua_State *L, int64_t value);
void   lua_pushuint(lua_State *L, unsigned int value);
void   lua_pushfloat(lua_State *L, float value);
void   lua_pushdouble(lua_State *L, double value);
void   lua_pushchar(lua_State *L, char value);

int lua_print_name_instance(lua_State *L);
int lua_print_name_class(lua_State *L);
int lua_concat_class_method(lua_State *L);

void check_table_fields(lua_State *L, int idx, ...);

#define LUABIND_TABLE_GETN(idx, var) \
  do { \
    var = (int)luaL_len(L,idx); \
  } while (0)

#define LUABIND_TABLE_TO_VECTOR(table, tipo, vector, longitud)	\
  do { \
    int luabind_len = (longitud); \
    int luabind_table = (table); \
    for(int luabind_i=0; luabind_i < luabind_len; luabind_i++) { \
      lua_rawgeti(L, luabind_table, luabind_i+1); \
      vector[luabind_i] = lua_to##tipo(L, -1); \
      lua_pop(L,1); \
    } \
  } while (0)

#define LUABIND_TABLE_TO_VECTOR_SUB1(table, tipo, vector, longitud)	\
  do { \
    int luabind_len = (longitud); \
    int luabind_table = (table); \
    for(int luabind_i=0; luabind_i < luabind_len; luabind_i++) { \
      lua_rawgeti(L, luabind_table, luabind_i+1); \
      vector[luabind_i] = lua_to##tipo(L, -1) - 1; \
      lua_pop(L,1); \
    } \
  } while (0)

#define LUABIND_VECTOR_TO_NEW_TABLE(tipo, vector, longitud) \
  do { \
    int luabind_len = longitud; \
    const tipo *luabind_vector = vector; \
    lua_createtable (L, luabind_len, 0); \
    for(int luabind_i=0; luabind_i < luabind_len; luabind_i++) { \
      lua_push##tipo(L,luabind_vector[luabind_i]); \
      lua_rawseti(L,-2,luabind_i+1); \
    } \
  } while (0)

#define LUABIND_FORWARD_CONTAINER_TO_NEW_TABLE(container_type, data_type, container) \
  do {									\
    const container_type *luabind_container = &container;		\
    lua_createtable (L, luabind_container->size(), 0);			\
    int luabind_index = 1;						\
    for (container_type::const_iterator luabind_i = luabind_container->begin(); \
	 luabind_i != luabind_container->end();				\
	 ++luabind_i) {							\
      lua_push##data_type(L, *luabind_i);				\
      lua_rawseti(L, -2, luabind_index++);				\
    }									\
  } while(0)


#define LUABIND_RETURN(type, value) \
  do { \
    lua_push##type(L, (value)); \
    ++luabind_num_returned_values; \
  } while (0)

#define LUABIND_RETURN_NIL() \
  do { \
    lua_pushnil(L); \
    ++luabind_num_returned_values; \
  } while (0)

#define LUABIND_RETURN_FROM_STACK(index) \
  do { \
    lua_pushvalue(L, index); \
    ++luabind_num_returned_values; \
  } while (0)

#define LUABIND_INCREASE_NUM_RETURNS(value) luabind_num_returned_values+=(value)


#endif // LUABINDUTIL_H

