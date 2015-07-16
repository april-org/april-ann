#include "luabindutil.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

#define PARENTS_REGISTRY_FIELD_NAME "luabind_parents"
#define CAST_REGISTRY_FIELD_NAME "luabind_cast"

void stackDump(lua_State *L) {
  int i;
  int top = lua_gettop(L);
  for (i = 1; i <= top; i++) {  /* repeat for each level */
    int t = lua_type(L, i);
    switch (t) {
    
    case LUA_TSTRING:  /* strings */
      fprintf(stderr,"`%s'", lua_tostring(L, i));
      break;
    
    case LUA_TBOOLEAN:  /* booleans */
      fprintf(stderr,lua_toboolean(L, i) ? "true" : "false");
      break;
    
    case LUA_TNUMBER:  /* numbers */
      fprintf(stderr,"%g", lua_tonumber(L, i));
      break;
    
    default:  /* other values */
      fprintf(stderr,"%s", lua_typename(L, t));
      break;
    
    }
    fprintf(stderr,"  ");  /* put a separator */
  }
  fprintf(stderr,"\n");  /* end the listing */
}

void makeWeakTable(lua_State *L) {
  if (!lua_getmetatable(L, -1)) {
    lua_newtable(L);
  }
  lua_pushstring(L, "__mode");
  lua_pushstring(L, "v");
  lua_rawset(L, -3);
  lua_setmetatable(L, -2);
}

void pushOrCreateTable(lua_State *L, int n, const char *field) {
  // stack:
  lua_getfield(L, n, field);
  // stack: [n].field
  if (lua_isnil(L, -1)) {
    // stack: nil
    lua_pop(L, 1);
    // stack:
    lua_newtable(L);
    // stack: table
    lua_pushvalue(L, -1);
    int top = lua_gettop(L);
    // stack: table table
    if (n < 0 && -n <= top) { // relative index from top
      lua_setfield(L, n - 2, field);
      // stack: table
    }
    else { // absolute index or pseudo-index
      lua_setfield(L, n, field);
      // stack: table
    }
  }
  // stack: [n].field
}

int isDerived(lua_State *L,
              const char *child,
              const char *parent) {
  int result = 0;
  // stack:
  pushOrCreateTable(L, LUA_REGISTRYINDEX, PARENTS_REGISTRY_FIELD_NAME);
  // stack: parents
  lua_pushstring(L, child);
  // stack: parents child_string
  lua_rawget(L, -2);
  // stack: parents child_table
  if (lua_isnil(L, -1)) {
    // stack: parents nil
    lua_pop(L, 2);
    // stack:
  }
  else {
    lua_pushstring(L, parent);
    // stack: parents child_table parent_string
    lua_rawget(L, -2);
    // stack: parents child_table value
    if (!lua_isnil(L, -1)) {
      result = 1;
    }
    // stack: parents child_table value
    lua_pop(L, 3);
    // stack:
  }
  return result;
}

void setParentsAtRegistry(lua_State *L,
                          const char *parent,
                          const char *child) {
  int top = lua_gettop(L);
  // stack:
  pushOrCreateTable(L, LUA_REGISTRYINDEX, PARENTS_REGISTRY_FIELD_NAME);
  // stack: parents
  pushOrCreateTable(L, -1, child);
  // stack: parents child_table
  lua_pushstring(L, parent);
  // stack: parents child_table parent_name
  while(lua_gettop(L) != top+2) { // while not empty stack
    // stack: ... parent_name
    lua_pushvalue(L, -1);
    // stack: ... parent_name parent_name
    lua_pushboolean(L, 1);
    // stack: ... parent_name parent_name true
    lua_rawset(L, top + 2);
    // stack: ... parent_name
    lua_rawget(L, top + 1);
    // stack: ... parent_table
    if (!lua_isnil(L, -1)) {
      int tbl = lua_gettop(L);
      lua_pushnil(L); // first key for lua_next
      // stack: ... parent_table nil
      while (lua_next(L, tbl) != 0) {
        // stack: ... parent_table ... key value
        lua_pop(L, 1);
        // stack: ... parent_table ... key
        lua_pushvalue(L, -1);
        // stack: ... parent_table ... key key
      }
      // stack: ... parent_table ...
      lua_remove(L, tbl);
      // stack: ... more names
    }
    else {
      lua_pop(L, 1);
      // stack: ...
    }
  }
  // stack: parents child_table
  lua_pop(L, 2);
  // stack:
}

void insertCast(lua_State *L, const char *derived, const char *base,
                int (*c_function)(lua_State *)) {
  // stack:
  pushOrCreateTable(L, LUA_REGISTRYINDEX, CAST_REGISTRY_FIELD_NAME);
  // stack: registry.luabind_cast
  pushOrCreateTable(L, -1, base);
  // stack: registry.luabind_cast registry.luabind_cast[base]
  lua_pushstring(L, derived);
  // stack: registry.luabind_cast registry.luabind_cast[base] "derived"
  lua_pushcfunction(L, c_function);
  // stack: registry.luabind_cast registry.luabind_cast[base] "derived" func
  lua_rawset(L, -3);
  // stack: registry.luabind_cast registry.luabind_cast[base]
  lua_pop(L, 2);
}

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

int lua_concat_class_method(lua_State *L) {
  int argn = lua_gettop(L); // number of arguments
  if (argn != 2) {
    lua_pushstring(L, "Incorrect number of arguments, expected 2");
    lua_error(L);
  }
  if (!lua_isstring(L,2)) {
    lua_pushstring(L, "Expected a string as second argument");
    lua_error(L);
  }
  if (!lua_istable(L,1)) {
    lua_pushstring(L, "Expected a class table as first argument");
    lua_error(L);    
  }
  const char *method_name = lua_tostring(L,2);
  // stack: class_table name
  lua_pushstring(L,"meta_instance");
  // stack: class_table name "meta_instance"
  lua_rawget(L,1);
  if (lua_isnil(L,-1)) {
    return 0;
  }
  // stack: class_table name meta_instance
  lua_pushstring(L,"__index");
  // stack: class_table name meta_instance "__index"
  lua_rawget(L,-2);
  // stack: class_table name meta_instance __index
  if (lua_isnil(L,-1)) {
    return 0;
  }
  // stack: class_table name meta_instance __index
  if (lua_istable(L,-1)) {
    // stack: class_table name meta_instance __index
    lua_pushstring(L,method_name);
    // stack: class_table name meta_instance __index "name"
    lua_rawget(L,-2);
  }
  else {
    // stack: class_table name meta_instance __index
    lua_pushnil(L);
    // stack: class_table name meta_instance __index nil
    lua_pushstring(L,method_name);
    // stack: class_table name meta_instance __index nil "name"
    lua_call(L,2,1);
  }
  // stack: class_table name meta_instance __index method
  if (lua_isnil(L,-1)) {
    return 0;
  }
  else {
    return 1;
  }
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

