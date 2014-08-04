/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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

#include "constString.h"
#include "stream.h"

namespace april_io {

  class OutputBufferStream : public Stream {
    char *buffer;
    char *pos;
    size_t expected_size;
  public:
    OutputBufferStream();
    virtual ~OutputBufferStream();
    virtual void close();
    virtual void flush();
    virtual bool isOpened() const;
    virtual bool eof();
    virtual int seek(long offset, int whence);
    virtual size_t read(void *ptr, size_t size, size_t nmemb);
    virtual size_t write(const void *ptr, size_t size, size_t nmemb);
    virtual void setExpectedSize(size_t sz);
    char *getBufferProperty();
    constString get() const;
  };

  class InputBufferStream : public Stream {
    constString buffer;
  public:
    InputBufferStream(const char *buffer, size_t n);
    virtual ~InputBufferStream();
    virtual void close();
    virtual void flush();
    virtual bool isOpened() const;
    virtual bool eof();
    virtual int seek(long offset, int whence);
    virtual size_t read(void *ptr, size_t size, size_t nmemb);
    virtual size_t write(const void *ptr, size_t size, size_t nmemb);
  };
  
} // namespace april_io
