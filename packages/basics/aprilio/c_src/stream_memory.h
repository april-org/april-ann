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
#ifndef STREAM_MEMORY_H
#define STREAM_MEMORY_H

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
#include <stdint.h> // for SIZE_MAX (using stdint.h because cstdint needs c++11
                    // support)

}
#include <cstdio>
#include <cstring>

#include "constString.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "unused_variable.h"

#define WRITE_ONLY_STREAM_MEMORY(name)                                  \
  virtual char operator[](size_t pos) const {                           \
    UNUSED_VARIABLE(pos);                                               \
    ERROR_EXIT(128, "Read only " #name "\n");                           \
    return 0;                                                           \
  }

#define READ_ONLY_STREAM_MEMORY(name)                                   \
  virtual char &operator[](size_t pos) {                                \
    UNUSED_VARIABLE(pos);                                               \
    ERROR_EXIT(128, "Write only " #name "\n");                          \
    return StreamMemory::DUMMY_CHAR;                                    \
  }                                                                     \
  virtual void clear() {                                                \
    ERROR_EXIT(128, "Read only " #name "\n");                           \
  }                                                                     \
  virtual void flush() {                                                \
    ERROR_EXIT(128, "Read only " #name "\n");                           \
  }

namespace AprilIO {

  class StreamMemory;

  /// Returns a whole line of the stream (a string delimited by \n).
  size_t extractLineFromStream(StreamInterface *source, StreamMemory *dest);
  
  /// Returns a whole line of the stream (a string delimited by \n), but
  /// avoiding lines which begins with #. Lines beginning with # are taken as
  /// commented lines.
  size_t extractULineFromStream(StreamInterface *source, StreamMemory *dest);

  ///////////////////////////////////////////////////////////////////////////
  
  class StreamMemory : public StreamBuffer {
  public:
    static const size_t BLOCK_SIZE = 1024;
    StreamMemory() { }
    virtual ~StreamMemory() { }
    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
    virtual char operator[](size_t pos) const = 0;
    virtual char &operator[](size_t pos) = 0;
    virtual void clear() = 0;
    virtual int push(lua_State *L) = 0;
  };

} // namespace AprilIO

#endif // STREAM_MEMORY_H
