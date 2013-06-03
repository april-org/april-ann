#include "error_print.h"
extern "C" {
#include "lua.h"
}

extern lua_State *globalL;

void print_lua_tracebak() {
  lua_pushstring(globalL, "Error at C++ code");
  lua_error(globalL);
  /*
    lua_getfield(globalL, LUA_GLOBALSINDEX, "debug");
    lua_getfield(globalL, -1, "traceback");
    lua_pushvalue(globalL, 1);
    lua_pushinteger(globalL, 2);
    lua_call(globalL, 2, 1);
    fprintf(stderr, "%s\n", lua_tostring(globalL, -1));
  */
}
