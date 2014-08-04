/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */

#include "lua_file.h"

namespace april_io {
  
  LuaFile::LuaFile(Stream *stream) : File(stream), Referenced() {
    if (!stream->isOpened()) {
      ERROR_EXIT(128, "Needs an opened stream\n");
    }
  }
  
  LuaFile::~LuaFile() { }
  
  int LuaFile::readAndPushNumberToLua(lua_State *L) {
    if (buffer_len == 0) return 0;
    constString token = getToken(" ,;\t\n\r");
    if (token.empty()) return 0;
    double number;
    if (!token.extract_double(&number))
      ERROR_EXIT(256, "Impossible to extract a number from current file pos\n");
    lua_pushnumber(L, number);
    return 1;
  }
    
  int LuaFile::readAndPushStringToLua(lua_State *L, int size) {
    if (buffer_len == 0) return 0;
    constString token = getToken(size);
    if (token.empty()) return 0;
    lua_pushlstring(L, (const char *)(token), token.len());
    return 1;
  }
  
  int LuaFile::readAndPushLineToLua(lua_State *L) {
    if (buffer_len == 0) return 0;
    constString line = extract_line();
    if (line.empty()) return 0;
    lua_pushlstring(L, (const char *)(line), line.len());
    return 1;
  }
  
  int LuaFile::readAndPushAllToLua(lua_State *L) {
    if (buffer_len == 0) return 0;
    constString line = getToken(1024);
    if (line.empty()) return 0;
    luaL_Buffer lua_buffer;
    luaL_buffinit(L, &lua_buffer);
    luaL_addlstring(&lua_buffer, (const char*)(line), line.len());
    while((line = getToken(1024)) && !line.empty()) {
      luaL_addlstring(&lua_buffer, (const char*)(line), line.len());
    }
    luaL_pushresult(&lua_buffer);
    return 1;
  }
  
  int LuaFile::readLua(lua_State *L) {
    if (!good()) {
      lua_pushnil(L);
      return 1;
    }
    /*
      "*n" reads a number; this is the only format that returns a number instead
      of a string.
      
      "*a" reads the whole file, starting at the current position. On end of
      file, it returns the empty string.
      
      "*l" reads the next line (skipping the end of line), returning nil on end
      of file. This is the default format.  number reads a string with up to
      that number of characters, returning nil on end of file. If number is
      zero, it reads nothing and returns an empty string, or nil on end of file.
    */
    int argn = lua_gettop(L); // number of arguments
    int num_returned_values=0;
    if (argn == 0) {
      num_returned_values += readAndPushLineToLua(L);
    }
    else {
      for (int i=1; i<=argn; ++i) {
        if (lua_isnil(L, i)) {
          num_returned_values += readAndPushLineToLua(L);
        }
        else if (lua_isnumber(L, i)) {
          int size = luaL_checkint(L, i);
          num_returned_values += readAndPushStringToLua(L, size);
        }
        else {
          const char *format = lua_tostring(L, i);
          // a number
          if (strcmp(format, "*n") == 0) {
            num_returned_values += readAndPushNumberToLua(L);
          }
          // the whole file
          else if (strcmp(format, "*a") == 0) {
            num_returned_values += readAndPushAllToLua(L);
          }
          // a line
          else if (strcmp(format, "*l") == 0) {
            num_returned_values += readAndPushLineToLua(L);
          }
          else {
            lua_pushfstring(L, "Unrecognized format string '%s'", format);
            lua_error(L);
          } // if (strcmp(format), ...) ...
        } // if isnil ... else if isnumber ... else ...
      } // for (int i=1; i<= argn; ++i)
    } // if (argn == 0) ... else
    return num_returned_values;
  }

  int LuaFile::seekLua(lua_State *L) {
    const char *whence = luaL_optstring(L, 1, "cur");
    int offset = luaL_optint(L, 2, 0);
    int int_whence;
    if (strcmp(whence, "cur") == 0) {
      int_whence = SEEK_CUR;
    }
    else if (strcmp(whence, "set") == 0) {
      int_whence = SEEK_SET;
    }
    else {
      int_whence = SEEK_END;
      lua_pushfstring(L, "Not supported whence '%s'", whence);
      lua_error(L);
    }
    int ret = seek(int_whence, offset);
    if (ret < 0) {
      lua_pushnil(L);
      lua_pushstring(L, "Impossible to execute seek");
      return 2;
    }
    else {
      lua_pushinteger(L, ret);
      return 1;
    }
  }

  int LuaFile::writeLua(lua_State *L) {
    int argn = lua_gettop(L); // number of arguments
    for (int i=1; i<=argn; ++i) {
      const char *value = luaL_checkstring(L, i);
      printf("%s", value);
    }
    return 0;
  }
  
  int LuaFile::flushLua(lua_State *L) {
    UNUSED_VARIABLE(L);
    flush();
    return 0;
  }

  int LuaFile::closeLua(lua_State *L) {
    UNUSED_VARIABLE(L);
    close();
    return 0;
  }

  int LuaFile::goodLua(lua_State *L) const {
    lua_pushboolean(L, good());
    return 1;
  }

} // namespace april_io
