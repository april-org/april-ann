#include "error_print.h"
extern "C" {
#include "lua.h"
}

extern lua_State *globalL;

void print_lua_tracebak() {
  lua_pushstring(globalL, "");
  lua_error(globalL);
}
