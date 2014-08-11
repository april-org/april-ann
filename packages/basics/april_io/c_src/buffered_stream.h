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
#ifndef BUFFERED_STREAM_H
#define BUFFERED_STREAM_H

#include <cstdio>
#include <cstring>

#include "constString.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "unused_variable.h"

namespace april_io {
  class BufferedStream : public Stream {
  public:
    
    BufferedStream(size_t buf_size = BUFSIZ);
    virtual ~BufferedStream();
    // virtual bool isOpened() const = 0;
    virtual void close();
    virtual off_t seek(int whence, int offset);
    virtual void flush();
    virtual int setvbuf(int mode, size_t size);
    // virtual bool hasError() const = 0;
    // virtual const char *getErrorMsg() const = 0;

  protected:

    virtual const char *nextInBuffer(size_t &buf_len);
    virtual char *nextOutBuffer(size_t &buf_len);
    
    /// Reads from the real stream and puts data into the given buffer.
    virtual ssize_t fillBuffer(char *dest, size_t max_size) = 0;
    /// Writes the given buffer into the real stream.
    virtual ssize_t flushBuffer(const char *source, size_t max_size) = 0;
    /// Closes the real stream.
    virtual void closeStream() = 0;
    /// Executes seek operation to the real stream.
    virtual off_t seekStream(int whence, int offset) = 0;
    /// Returns true if the real stream is at EOF.
    virtual bool eofStream() const = 0;
  
  private:
    /// Reading buffer
    char *in_buffer;
    /// Writing buffer
    char *out_buffer;
    /// Reserved size of the buffer
    size_t max_buffer_len;
  };
  
} // namespace april_io

#endif // BUFFERED_STREAM_H
