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
#include "april_assert.h"
#include "lua_string.h"

namespace april_io {
  OutputLuaStringStream::OutputLuaStringStream(lua_State *L,
                                               size_t block_size) :
    L(L),
    block_size(block_size), total_size(0), closed(false) {
    luaL_buffinit(L, &lua_buffer);
    out_buffer = new char[block_size];
  }
  
  OutputLuaStringStream::~OutputLuaStringStream() {
    close();
  }
  
  bool OutputLuaStringStream::empty() const {
    return total_size == 0u;
  }

  size_t OutputLuaStringStream::size() const {
    return total_size;
  }

  char &OutputLuaStringStream::operator[](size_t pos) {
    UNUSED_VARIABLE(pos);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return StreamInterface::DUMMY_CHAR;
  }
  
  void OutputLuaStringStream::clear() {
    if (total_size > 0) {
      close();
      closed = false;
      luaL_buffinit(L, &lua_buffer);
      resetBuffers();
      total_size = 0;
    }
  }
  
  bool OutputLuaStringStream::isOpened() const {
    return !closed;
  }
  
  void OutputLuaStringStream::close() {
    if (out_buffer != 0) {
      flush();
      closed = true;
    }
  }
  
  int OutputLuaStringStream::push(lua_State *L) {
    if (L != this->L) {
      ERROR_EXIT(128, "Incorrect lua_State reference\n");
    }
    flush();
    UNUSED_VARIABLE(L);
    luaL_pushresult(&lua_buffer);
    return 1;
  }
  
  void OutputLuaStringStream::flush() {
    if (getOutBufferPos() > 0) {
      total_size += getOutBufferPos();
      luaL_addlstring(&lua_buffer, out_buffer, getOutBufferPos());
      resetOutBuffer();
    }
  }
  
  int OutputLuaStringStream::setvbuf(int mode, size_t size) {
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
  
  bool OutputLuaStringStream::hasError() const {
    return false;
  }
  
  const char *OutputLuaStringStream::getErrorMsg() const {
    return StreamInterface::NO_ERROR_STRING;
  }
  
  const char *OutputLuaStringStream::nextInBuffer(size_t &buf_len) {
    buf_len=0;
    return 0;
  }
  
  char *OutputLuaStringStream::nextOutBuffer(size_t &buf_len) {
    buf_len = block_size - getOutBufferPos();
    return out_buffer + getOutBufferPos();
  }
  
  bool OutputLuaStringStream::eofStream() const {
    return !isOpened();
  }
  
  void OutputLuaStringStream::moveOutBuffer(size_t len) {
    Stream::moveOutBuffer(len);
    flush();
  }
  
  ///////////////////////////////////////////////////////////////////////////
  
  InputLuaStringStream::InputLuaStringStream(lua_State *L, int pos) :
    L(L), data_pos(0) {
    if (!lua_isstring(L, pos)) {
      ERROR_EXIT(256, "Needs a Lua string argument\n");
    }
    total_size = static_cast<size_t>(luaL_len(L, pos));
    lua_pushvalue(L, pos);
    ref = luaL_ref(L, LUA_REGISTRYINDEX);
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
    data = lua_tostring(L, -1);
    lua_pop(L, -1);
  }
  
  InputLuaStringStream::~InputLuaStringStream() {
    close();
  }
  
  bool InputLuaStringStream::empty() const {
    return total_size == 0u;
  }
  
  size_t InputLuaStringStream::size() const {
    return total_size;
  }
  
  char InputLuaStringStream::operator[](size_t pos) const {
    april_assert(pos < total_size);
    return data[pos];
  }

  int InputLuaStringStream::push(lua_State *L) {
    if (L != this->L) {
      ERROR_EXIT(128, "Incorrect lua_State reference\n");
    }
    lua_pushlstring(L, data, total_size);
    return 1;
  }
  
  bool InputLuaStringStream::isOpened() const {
    return ref != LUA_NOREF;
  }
  
  void InputLuaStringStream::close() {
    if (ref != LUA_NOREF) {
      luaL_unref(L, LUA_REGISTRYINDEX, ref);
      ref = LUA_NOREF;
      data = 0;
      total_size = 0;
    }
  }
  
  off_t InputLuaStringStream::seek(int whence, int offset) {
    switch(whence) {
    case SEEK_SET:
      data_pos = offset;
      break;
    case SEEK_CUR:
      data_pos = data_pos + getInBufferPos() + offset;
      break;
    case SEEK_END:
      data_pos = total_size - offset;
      break;
    }
    resetBuffers();
    return data_pos;
  }
  
  int InputLuaStringStream::setvbuf(int mode, size_t size) {
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
  
  bool InputLuaStringStream::hasError() const {
    return false;
  }
  
  const char *InputLuaStringStream::getErrorMsg() const {
    return StreamInterface::NO_ERROR_STRING;
  }
  
  const char *InputLuaStringStream::nextInBuffer(size_t &buf_len) {
    data_pos = data_pos + getInBufferPos();
    april_assert(data_pos <= total_size);
    buf_len = total_size - data_pos;
    return data + data_pos;
  }

  char *InputLuaStringStream::nextOutBuffer(size_t &buf_len) {
    buf_len = 0;
    return 0;
  }
  
  bool InputLuaStringStream::eofStream() const {
    return data_pos + getInBufferPos() >= total_size;
  }
} // namespace april_io
