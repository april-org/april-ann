#include "luabindutil.h"
#include <cstring>
#include <cstdarg>

bool lua_isFILE(lua_State *L, int idx) {
  void *ud;
  luaL_checkany(L, idx);
  ud = lua_touserdata(L, idx);
  lua_getfield(L, LUA_REGISTRYINDEX, LUA_FILEHANDLE);
  if (ud == NULL || !lua_getmetatable(L, 1) || !lua_rawequal(L, -2, -1))
    return false;  /* not a file */
  else if (*((FILE **)ud) == NULL)
    return false;
  else
    return true;
}

FILE*  lua_toFILE(lua_State *L, int idx) {
  void *ud;
  luaL_checkany(L, idx);
  ud = lua_touserdata(L, idx);
  lua_getfield(L, LUA_REGISTRYINDEX, LUA_FILEHANDLE);
  if (ud == NULL || !lua_getmetatable(L, 1) || !lua_rawequal(L, -2, -1))
    return 0;  /* not a file */
  return *((FILE **)ud);
}


int equaluserdata(lua_State *L) {
  void **ptra = (void**)lua_touserdata(L,1);
  void **ptrb = (void**)lua_touserdata(L,2);
  lua_pushboolean(L, (*ptra == *ptrb));
  return 1;
}

bool lua_tobool(lua_State *L, int idx) {
  return (bool)(lua_toboolean(L, idx));
}

int lua_toint(lua_State *L, int idx) {
  // FIXME: Considerar si vale la pena comprobar si el numero
  // leido es realmente un entero
  return int(lua_tonumber(L, idx));
}

int64_t lua_toint64(lua_State *L, int idx) {
  // FIXME: Considerar si vale la pena comprobar si el numero
  // leido es realmente un entero
  return int64_t(lua_tonumber(L, idx));
}

unsigned int lua_touint(lua_State *L, int idx) {
  // FIXME: Considerar si vale la pena comprobar si el numero
  // leido es realmente un entero
  return (unsigned int)(lua_tonumber(L, idx));
}

float lua_tofloat(lua_State *L, int idx) {
  return float(lua_tonumber(L, idx));
}

double lua_todouble(lua_State *L, int idx) {
  return double(lua_tonumber(L, idx));
}

char *lua_toNEW_STRING(lua_State *L, int idx) {
  int len = lua_strlen(L, idx);
  char *buf = new char[len+1];
  // Segun el manual de lua, el resultado de lua_tostring siempre lleva
  // un '\0' al final, asi que es seguro copiar len+1 caracteres
  memcpy(buf, lua_tostring(L, idx), len+1);
  return buf;
}

char lua_tochar(lua_State *L, int idx) {
  // FIXME: Considerar si vale la pena comprobar que la cadena
  // consta realmente de un unico caracter
  return lua_tostring(L, idx)[0];
}


void *lua_tolightuserdata(lua_State *L, int idx) {
  return lua_touserdata(L, idx);
}

bool lua_isbool(lua_State *L, int idx) {
  return lua_isboolean(L,idx);
}

bool lua_isint(lua_State *L, int idx) {
  if(!lua_isnumber(L, idx)) return false;
  else {
    double d = lua_tonumber(L, idx);
    return (d == int(d));
  }
}

bool lua_isint64(lua_State *L, int idx) {
  if(!lua_isnumber(L, idx)) return false;
  else {
    double d = lua_tonumber(L, idx);
    return (d == int64_t(d));
  }
}

bool lua_isuint(lua_State *L, int idx) {
  if(!lua_isnumber(L, idx)) return false;
  else {
    double d = lua_tonumber(L, idx);
    return (d == (unsigned int)(d));
  }
}

bool lua_isfloat(lua_State *L, int idx) {
  return lua_isnumber(L, idx);
}

bool lua_isdouble(lua_State *L, int idx) {
  return lua_isnumber(L, idx);
}

bool lua_ischar(lua_State *L, int idx) {
  return (lua_isstring(L, idx) && (lua_strlen(L, idx) == 1));
}

bool lua_isNEW_STRING(lua_State *L, int idx) {
  return lua_isstring(L, idx);
}

bool lua_isconstString(lua_State *L, int idx) {
  return lua_isstring(L, idx);
}

void lua_pushbool(lua_State *L, bool value) {
  lua_pushboolean(L, value);
}

void lua_pushint(lua_State *L, int value) {
  lua_pushnumber(L, value);
}

void lua_pushint(lua_State *L, int64_t value) {
  lua_pushnumber(L, value);
}

void lua_pushuint(lua_State *L, unsigned int value) {
  lua_pushnumber(L, value);
}

void lua_pushfloat(lua_State *L, float value) {
  lua_pushnumber(L, value);
}

void lua_pushdouble(lua_State *L, double value) {
  lua_pushnumber(L, value);
}

void lua_pushchar(lua_State *L, char value) {
  lua_pushlstring(L, &value, 1);
}

int lua_print_name_instance(lua_State *L) {
  char buff[128];
  if (lua_istable(L,1)) {
    sprintf(buff,"this instance is a table (%p) not a userdata",
	    lua_topointer(L,1));
  } else {
    void **ptr = (void**)lua_touserdata(L,1);
    if (!lua_getmetatable(L,1)) {
      sprintf(buff,"unknown object %p",(void*)ptr);
    } else {
      lua_pushstring(L,"id");
      lua_rawget(L,-2);
      const char *class_name = lua_tostring(L,-1);
      if (class_name == 0) {
	sprintf(buff,"unknown object %p",(void*)ptr);
      } else {
	sprintf(buff,"instance %p of %s",*ptr,class_name);
      }
      lua_pop(L,2);
    }
  }
  lua_pushstring(L, buff);
  return 1;
}

int lua_print_name_class(lua_State *L) {
  char buff[128];
  sprintf(buff,"unknown class");
  if (lua_getmetatable(L,1)) {
    lua_pushstring(L,"id");
    lua_rawget(L,-2);
    const char *class_name = lua_tostring(L,-1);
    if (class_name) {
      sprintf(buff,"class %s",class_name);
    }
    lua_pop(L,2);
  }
  lua_pushstring(L, buff);
  return 1;
}

void check_table_fields(lua_State *L, int idx, ...) {
  lua_pushnil(L);
  while(lua_next(L, idx) != 0) {
    // Clave en -2, valor en -1
    lua_pop(L, 1); // el valor no interesa, clave en -1
    const char *key = lua_tostring(L, -1);
    va_list ap;
    va_start(ap, idx);
    const char *word = va_arg(ap, const char *);
    bool found = false;
    while (word != 0) {
      if (strcmp(word, key) == 0) {
        found = true;
        break;
      }

      word = va_arg(ap, const char *);
    }
    va_end(ap);

    if (!found) {
      lua_pushfstring(L, "ERROR: field %s not allowed in table", key);
      lua_error(L);
    }
  }
}

