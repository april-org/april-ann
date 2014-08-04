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
#ifndef LUA_FILE_H
#define LUA_FILE_H

#include <cstdio>
#include <cstring>

#include "file.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "unused_variable.h"

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

namespace april_io {
  
  /**
   * The class LuaFile is exported to Lua and has an specific readLua method, a
   * writeLua and a seekLua whome take their arguments from Lua stack and push
   * them there results also there.
   */
  class LuaFile : protected File, public Referenced {
  private:
    /** Lua methods: is more efficient to directly put strings at the Lua stack,
        instead of build a C string and return it. Pushing to the stack
        approximately doubles the memory because needs the original object and the
        Lua string. Returning a C string needs a peak of three times more memory,
        because it needs the original object, the C string, and finally the Lua
        string, even if the C string is removed after written to Lua. **/
    /// Reads a number from the memory and pushes it to the Lua stack
    int readAndPushNumberToLua(lua_State *L);
    //  int readAndPushCharToLua(lua_State *L);
    /// Reads a string of maximum given size from the memory and pushes it to the Lua stack
    int readAndPushStringToLua(lua_State *L, int size);
    /// Reads a line from the memory and pushes it to the Lua stack
    int readAndPushLineToLua(lua_State *L);
    /// Reads the whole memory and pushes it to the Lua stack
    int readAndPushAllToLua(lua_State *L);
    /*****************/
    
  public:

    LuaFile(Stream *stream);
    virtual ~LuaFile();
    
    int readLua(lua_State *L);
    int writeLua(lua_State *L);
    int seekLua(lua_State *L);
    int flushLua(lua_State *L);
    int closeLua(lua_State *L);
    int goodLua(lua_State *L) const;

  };
} // namespace april_io

#endif // LUA_FILE_H
