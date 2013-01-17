#undef LUABIND_CHECK_ARGN
#define LUABIND_CHECK_ARGN(op, n) \
  do { \
    int luabind_n = (n); \
    int argn = lua_gettop(L); \
    if (!(argn op luabind_n)) { \
      lua_pushfstring(L, FUNCTION_NAME "() requires " #op " " #n " arguments" \
          "(%d given)", argn); \
      lua_error(L); \
    } \
  } while(0)

#undef LUABIND_CHECK_PARAMETER
#define LUABIND_CHECK_PARAMETER(i, type) \
  do { \
    int luabind_pos = (i); \
    if (!lua_is##type(L, luabind_pos)) { \
      lua_pushfstring(L, FUNCTION_NAME "() requires a %s as its " #i "%s parameter", \
        #type, (luabind_pos==1 ? "st" : (luabind_pos==2 ? "nd" : (luabind_pos==3 ? "rd" : "th")))); \
      lua_error(L); \
    } \
  } while(0)


#undef LUABIND_GET_PARAMETER
#define LUABIND_GET_PARAMETER(i, type, var) \
  do { \
    int luabind_pos = (i); \
    if (!lua_is##type(L, luabind_pos)) { \
      lua_pushfstring(L, FUNCTION_NAME "() requires a %s as its " #i "%s parameter", \
        #type, (luabind_pos==1 ? "st" : (luabind_pos==2 ? "nd" : (luabind_pos==3 ? "rd" : "th")))); \
      lua_error(L); \
    } \
    var = lua_to##type(L, luabind_pos); \
  } while(0)

#undef LUABIND_GET_OPTIONAL_PARAMETER
#define LUABIND_GET_OPTIONAL_PARAMETER(i, type, var, default_value) \
  do { \
    int luabind_pos = (i); \
    if (lua_type(L, luabind_pos) == LUA_TNONE) var = default_value; \
    else { \
      if (!lua_is##type(L, luabind_pos)) { \
       lua_pushfstring(L, FUNCTION_NAME "() requires a %s as its %d%s parameter", \
          #type, luabind_pos, (luabind_pos==1 ? "st" : (luabind_pos==2 ? "nd" : (luabind_pos==3 ? "rd" : "th")))); \
        lua_error(L); \
      } \
      var = lua_to##type(L, luabind_pos); \
    } \
  } while(0)

#undef LUABIND_GET_TABLE_PARAMETER 
#define LUABIND_GET_TABLE_PARAMETER(i, name, type, var) \
  do { \
    int luabind_pos_table = (i); \
    lua_pushstring(L, #name); \
    lua_rawget(L, luabind_pos_table); \
    if (!lua_is##type(L,-1)) { \
      lua_pushfstring(L, FUNCTION_NAME "() requires a \"%s\" field with type \"%s\" " \
          "in its " #i "%s parameter", #name, #type, \
          (luabind_pos_table==1 ? "st" : (luabind_pos_table==2 ? "nd" : (luabind_pos_table==3 ? "rd" : "th")))); \
      lua_error(L); \
    } \
    var = lua_to##type(L, -1); \
    lua_pop(L, 1); \
  } while (0)

#undef LUABIND_GET_TABLE_OPTIONAL_PARAMETER
#define LUABIND_GET_TABLE_OPTIONAL_PARAMETER(i, name, type, var, default_value) \
  do { \
    int luabind_pos_table = (i); \
    lua_pushstring(L, #name); \
    lua_rawget(L, luabind_pos_table); \
    if (lua_isnil(L, -1)) var = default_value; \
    else { \
      if (!lua_is##type(L,-1)) { \
        lua_pushfstring(L, FUNCTION_NAME "() requires a \"%s\" field " \
            "with type \"%s\" in its " #i "%s parameter", #name, #type, \
            (luabind_pos_table==1 ? "st" : (luabind_pos_table==2 ? "nd" : (luabind_pos_table==3 ? "rd" : "th")))); \
        lua_error(L); \
      } \
      var = lua_to##type(L, -1); \
    } \
    lua_pop(L, 1); \
  } while (0)

#undef LUABIND_ERROR
#define LUABIND_ERROR(str) \
  do {\
    lua_pushstring(L, FUNCTION_NAME ": " str); \
    lua_error(L); \
  } while (0)

#undef LUABIND_FERROR1
#define LUABIND_FERROR1(str, p1) \
  do {\
    lua_pushfstring(L, FUNCTION_NAME ": " str, p1); \
    lua_error(L); \
  } while (0)

#undef LUABIND_FERROR2
#define LUABIND_FERROR2(str, p1, p2) \
  do {\
    lua_pushfstring(L, FUNCTION_NAME ": " str, p1, p2); \
    lua_error(L); \
  } while (0)

#undef LUABIND_FERROR3
#define LUABIND_FERROR3(str, p1, p2, p3) \
  do {\
    lua_pushfstring(L, FUNCTION_NAME ": " str, p1, p2, p3); \
    lua_error(L); \
  } while (0)

#undef LUABIND_FERROR4
#define LUABIND_FERROR4(str, p1, p2, p3, p4)	\
  do {\
    lua_pushfstring(L, FUNCTION_NAME ": " str, p1, p2, p3, p4);	\
    lua_error(L); \
  } while (0)

#undef LUABIND_FERROR5
#define LUABIND_FERROR5(str, p1, p2, p3, p4, p5)	\
  do {\
    lua_pushfstring(L, FUNCTION_NAME ": " str, p1, p2, p3, p4, p5);	\
    lua_error(L); \
  } while (0)

