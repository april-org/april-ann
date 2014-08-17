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
#ifndef LUA_STRING_H
#define LUA_STRING_H

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}
#include "stream_memory.h"

namespace AprilIO {
  
  /**
   * A class which generates a Lua string on-the-fly using small intermediate C
   * strings. FIXME: This class CANNOT BE EXPORTED TO Lua because it needs the
   * Lua stack with the same content as when the object was created.
   */
  class OutputLuaStringStream : public StreamMemory {
    lua_State *L;
    const size_t block_size;
    size_t total_size;
    luaL_Buffer lua_buffer;
    char *out_buffer;
    bool closed;

  public:
    OutputLuaStringStream(lua_State *L,
                          size_t block_size=StreamMemory::BLOCK_SIZE);
    virtual ~OutputLuaStringStream();
    
    WRITE_ONLY_STREAM(OutputLuaStringStream);
    WRITE_ONLY_STREAM_MEMORY(OutputLuaStringStream);

    virtual bool empty() const;
    virtual size_t size() const;
    virtual char &operator[](size_t pos);
    virtual void clear();
    virtual int push(lua_State *L);

    virtual bool isOpened() const;
    virtual void close();
    virtual off_t seek(int whence = SEEK_CUR, int offset = 0);
    virtual void flush();
    virtual int setvbuf(int mode, size_t size);
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
    
  protected:
    virtual const char *nextInBuffer(size_t &buf_len);
    virtual char *nextOutBuffer(size_t &buf_len);
    virtual bool eofStream() const;

    void moveOutBuffer(size_t len);
  };

  /**
   * A class which reads from a Lua string.
   */
  class InputLuaStringStream : public StreamMemory {
    lua_State *L;
    int ref;
    const char *data;
    size_t total_size, data_pos;
  
  public:
    InputLuaStringStream(lua_State *L, int pos);
    virtual ~InputLuaStringStream();
    
    READ_ONLY_STREAM(InputLuaStringStream);
    READ_ONLY_STREAM_MEMORY(InputLuaStringStream);
    virtual bool empty() const;
    virtual size_t size() const;
    virtual char operator[](size_t pos) const;
    virtual int push(lua_State *L);
    
    virtual bool isOpened() const;
    virtual void close();
    virtual off_t seek(int whence = SEEK_CUR, int offset = 0);
    virtual int setvbuf(int mode, size_t size);
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
    
  protected:
    virtual const char *nextInBuffer(size_t &buf_len);
    virtual char *nextOutBuffer(size_t &buf_len);
    virtual bool eofStream() const;
  };

} // namespace AprilIO

#endif // LUA_STRING_H
