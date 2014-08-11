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
#ifndef CSTRING_STREAM_H
#define CSTRING_STREAM_H

extern "C" {
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

namespace april_io {

  class CStringStream : public StreamMemory {
  public:
    
    CStringStream(size_t block_size = StreamMemory::BLOCK_SIZE,
                  size_t max_size = StreamMemory::MAX_LIMIT_SIZE);
    virtual ~CStringStream();
    // virtual bool isOpened() const = 0;
    // virtual void close() = 0;
    virtual off_t seek(int whence, int offset);
    virtual void flush();
    virtual int setvbuf(int mode, size_t size);
    // virtual bool hasError() const = 0;
    // virtual const char *getErrorMsg() const = 0;
    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
    virtual char operator[](size_t pos) const = 0;
    virtual char &operator[](size_t pos) = 0;
    virtual void clear() = 0;
    virtual int push(lua_State *L) = 0;
    
  protected:
    
    /// Closes the real stream.
    virtual void closeStream() = 0;
    /// Executes seek operation to the real stream.
    virtual off_t seekStream(int whence, int offset) = 0;
  
  private:
    char *str;
    size_t capacity, in_pos, out_pos;
    
    // Auxiliary private methods
    size_t getInBufferAvailableSize() const {
      return max_size - in_pos;
    }
    const char *getInBuffer(size_t &buffer_len, size_t max_size,
                            const char *delim) {
      buffer_len = april_utils::min(capacity - in_pos, max_size);
      return str + in_pos;
    }
    char *getOutBuffer(size_t &memory_len, size_t max_size) {
    }
    void moveInBuffer(size_t len) {
    }
    void moveOutBuffer(size_t len) {
    }
  };

} // namespace april_io

#endif // CSTRING_STREAM_H
