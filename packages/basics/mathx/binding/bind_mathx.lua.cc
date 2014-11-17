//BIND_HEADER_C
#include <cstdlib>
#include <cstdio>
extern "C" {
  extern int luaopen_mathx(lua_State *L);
}
//BIND_END

//BIND_STATIC_CONSTRUCTOR require_mathx
{
  if (!luaopen_mathx(L)) {
    fprintf(stderr, "Unable to open mathx library\n");
    exit(128);
  }
  lua_pop(L,1);
}
//BIND_END
