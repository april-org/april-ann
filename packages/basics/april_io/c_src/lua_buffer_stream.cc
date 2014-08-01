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
#include <cstring>

#include "error_print.h"
#include "lua_buffer_stream.h"
#include "unused_variable.h"

namespace april_io {
        
  LuaOutputBufferStream::LuaOutputBufferStream(lua_State *L) :
    buffer_ptr(0), total_bytes(0), expected_size(0) {
    luaL_buffinit(L, &lua_buffer);
  }
        
  LuaOutputBufferStream::~LuaOutputBufferStream() {
    close();
  }

  void LuaOutputBufferStream::close() {
    if (buffer_ptr != 0) {
      luaL_addsize(&lua_buffer, total_bytes);
      buffer_ptr = 0;
    }
  }

  int LuaOutputBufferStream::push() {
    luaL_pushresult(&lua_buffer);
    return 1;
  }
  
  void LuaOutputBufferStream::flush() { }
        
  bool LuaOutputBufferStream::isOpened() const { return buffer_ptr != 0; }
  
  bool LuaOutputBufferStream::eof() { return total_bytes == expected_size; }
        
  int LuaOutputBufferStream::seek(long offset, int whence) {
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(whence);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
  
  size_t LuaOutputBufferStream::read(void *ptr, size_t size, size_t nmemb) {
    UNUSED_VARIABLE(ptr);
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(nmemb);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
        
  size_t LuaOutputBufferStream::write(const void *ptr, size_t size, size_t nmemb) {
    size_t len = size*nmemb;
    if (len + total_bytes > expected_size) {
      ERROR_EXIT(128, "Buffer overflow\n");
    }
    memcpy(buffer_ptr, ptr, len);
    buffer_ptr  += len;
    total_bytes += len;
    return len;
  }

  void LuaOutputBufferStream::setExpectedSize(size_t sz) {
    expected_size = sz;
    buffer_ptr = luaL_prepbuffsize(&lua_buffer, sz);
    if (buffer_ptr == 0) ERROR_EXIT(256, "Impossible to get the buffer\n");
  }
  
  ////////////////////////////////////////////////////////////////////////////

  LuaInputBufferStream::LuaInputBufferStream(lua_State *L) :
    L(L), pos(0) {
    if (!lua_isstring(L,1)) {
      ERROR_EXIT(256, "Needs a Lua string passed as 1st argument\n");
    }
    registry_index = new char[19];
    snprintf(registry_index, 18, "%p", this);
    total_len = static_cast<size_t>(luaL_len(L,1));
    data = lua_tostring(L, 1);
    lua_pushvalue(L, 1);
    lua_setfield(L, LUA_REGISTRYINDEX, registry_index);
  }
        
  LuaInputBufferStream::~LuaInputBufferStream() {
    close();
  }

  void LuaInputBufferStream::close() {
    if (registry_index != 0) {
      lua_pushnil(L);
      lua_setfield(L, LUA_REGISTRYINDEX, registry_index);
      delete[] registry_index;
      registry_index = 0;
    }
  }
  
  void LuaInputBufferStream::flush() { }
        
  bool LuaInputBufferStream::isOpened() const { return registry_index != 0; }
  
  bool LuaInputBufferStream::eof() { return pos == total_len; }
        
  int LuaInputBufferStream::seek(long offset, int whence) {
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(whence);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
  
  size_t LuaInputBufferStream::read(void *ptr, size_t size, size_t nmemb) {
    size_t len = size * nmemb;
    if (len > total_len - pos) len = total_len - pos;
    memcpy(ptr, data+pos, len);
    pos += len;
    return len;
  }
        
  size_t LuaInputBufferStream::write(const void *ptr, size_t size, size_t nmemb) {
    UNUSED_VARIABLE(ptr);
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(nmemb);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
  
} // namespace april_io
