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
#ifndef STREAM_H
#define STREAM_H
#include <cstdlib>

#include "referenced.h"
#include "unused_variable.h"

namespace april_io {

  /**
   * Defines the basic interface of data streams.
   *
   * The class Stream defines the basic interface for opening, closing and
   * interacting with a generic Stream of data. It does decalre read and write
   * methods for input/output operations.
   */
  class Stream : public Referenced {
  public:
    Stream() : Referenced() { }
    virtual ~Stream() { }
    virtual void close() = 0;
    virtual void flush() = 0;
    virtual bool isOpened() const = 0;
    virtual bool eof() = 0;
    virtual int seek(long offset, int whence) = 0;
    virtual size_t read(void *ptr, size_t size, size_t nmemb) = 0;
    virtual size_t write(const void *ptr, size_t size, size_t nmemb) = 0;
    /// Some objects needs to know the expected size before begin to write
    /// things, so this method is where this size is given.
    virtual void setExpectedSize(size_t sz) {
      UNUSED_VARIABLE(sz);
    }
  };
  
} // namespace april_io
#endif // STREAM_H
