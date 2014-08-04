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

  class StreamMemory;

  /// Returns a whole line of the stream (a string delimited by \n).
  size_t extractLineFromStream(Stream *source, StreamMemory *dest);
  
  /// Returns a whole line of the stream (a string delimited by \n), but
  /// avoiding lines which begins with #. Lines beginning with # are taken as
  /// commented lines.
  size_t extractULineFromStream(Stream *source, StreamMemory *dest);

  ///////////////////////////////////////////////////////////////////////////
  
  class StreamMemory : public Stream {
  public:
    
    static const size_t MAX_LIMIT_SIZE = SIZE_MAX; // from cstdint
    static const size_t BLOCK_SIZE = 1024;
    
    StreamMemory(size_t block_size = BLOCK_SIZE,
                 size_t max_size = MAX_LIMIT_SIZE);
    virtual ~StreamMemory();
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
    /// Size of every Lua addlstring block
    size_t block_size;
    /// Maximum size of the allocated buffer
    size_t max_size;
    /// Current input buffer block pointer.
    const char *in_block;
    /// Length of the current input block pointer.
    size_t in_block_len;
    /// Current output buffer block pointer.
    char *out_block;
    /// Length of the current output block pointer.
    size_t out_block_len;

    // Auxiliary private methods
    size_t getInBufferAvailableSize() const;
    const char *getInBuffer(size_t &memory_len, size_t max_size,
                            const char *delim);
    char *getOutBuffer(size_t &memory_len, size_t max_size);
    void moveInBuffer(size_t len);
    void moveOutBuffer(size_t len);
  };

} // namespace april_io

#endif // STREAM_MEMORY_H
