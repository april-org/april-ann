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
#ifndef C_STRING_H
#define C_STRING_H

#include "mystring.h"
#include "stream_memory.h"

namespace april_io {
  
  class CStringStream : public StreamMemory {
    april_utils::string data;
    size_t in_pos, out_pos;
    
  public:
    CStringStream();
    CStringStream(const april_utils::string &str);
    CStringStream(const char *str, size_t size);
    virtual ~CStringStream();
    
    constString getConstString() const;

    virtual bool empty() const;
    virtual size_t size() const;
    virtual char operator[](size_t pos) const;
    virtual char &operator[](size_t pos);
    virtual void clear();
    virtual int push(lua_State *L);

    virtual bool isOpened() const;
    virtual void close();
    virtual off_t seek(int whence, int offset);
    virtual void flush();
    virtual int setvbuf(int mode, size_t size);
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
    
  protected:
    virtual const char *nextInBuffer(size_t &buf_len);
    virtual char *nextOutBuffer(size_t &buf_len);
    virtual bool eofStream() const;
  };
  
} // namespace april_io

#endif // LUA_STRING_H
