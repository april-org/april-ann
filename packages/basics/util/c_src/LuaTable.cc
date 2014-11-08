#include "lua_table.h"

namespace AprilUtils {

  LuaTable::LuaTable(lua_State *L) {
    lua_newtable(L);
    init(L, -1);
    lua_pop(L,1);
  }
  
  LuaTable::LuaTable(lua_State *L, int i) {
    init(L,i);
  }

  LuaTable::LuaTable(const LuaTable &other) : L(other.L), ref(other.ref) {
  }

  LuaTable::~LuaTable() {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
  }

  LuaTable &LuaTable::operator=(const LuaTable &other) {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
    L = other.L;
    ref = other.ref;
  }
  
  void LuaTableOptions::init(lua_State *L, int i) {
    this->L = L;
    if (lua_isnil(L,i) || lua_type(L,i) == LUA_TNONE) ref = LUA_NOREF;
    else {
      lua_pushvalue(L,i);
      if (!lua_istable(L,-1)) {
        ERROR_EXIT1(128,"Expected a table parameter at pos %d\n", i);
      }
      this->ref = luaL_ref(L, LUA_REGISTRYINDEX);
    }
  }
  
  void LuaTable::checkAndGetRef() {
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Incorrect Lua reference\n");
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
  }
  
  template<>
  static int convertTo<int>(lua_State *L) {
    return lua_tointeger(L,-1);
  }

  template<>
  static float convertTo<float>(lua_State *L) {
    return static_cast<float>(lua_tonumber(L,-1));
  }

  template<>
  static double convertTo<double>(lua_State *L) {
    return lua_tonumber(L,-1);
  }
  
  template<>
  static bool convertTo<bool>(lua_State *L) {
    return lua_toboolean(L, -1);
  }

  template<>
  static string convertTo<string>(lua_State *L) {
    string aux(lua_tostring(L, -1), lua_len(L, -1));
    return aux;
  }
  
  template<>
  static void pushInto<int>(lua_State *L, int value) {
    lua_pushnumber(L, static_cast<double>(value));
  }

  template<>
  static void pushInto<float>(lua_State *L, float value) {
    lua_pushnumber(L, static_cast<double>(value));
  }

  template<>
  static void pushInto<double>(lua_State *L, double value) {
    lua_pushnumber(L, value);
  }

  template<>
  static void pushInto<bool>(lua_State *L, bool value) {
    lua_pushboolean(L, value);
  }

  template<>
  static void pushInto<string>(lua_State *L, const string &value) {
    lua_pushlstring(L, value.c_str(), value.size());
  }

}
