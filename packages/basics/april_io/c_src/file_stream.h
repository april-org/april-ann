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
#ifndef FILE_STREAM_H
#define FILE_STREAM_H

#include <cstdio>
#include <cstring>

#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "buffered_stream.h"
#include "unused_variable.h"

namespace april_io {
  class FileStream : public BufferedStream {
    /// File descriptor.
    int fd, errnum, flags;
    bool is_eof;
    
    template<typename T>
    T checkReturnValue(T ret_value);

  protected:
    virtual ssize_t fillBuffer(char *dest, size_t max_size);
    virtual ssize_t flushBuffer(const char *source, size_t max_size);
    virtual void closeStream();
    virtual off_t seekStream(int whence, int offset);
    virtual bool eofStream() const;

  public:
    
    FileStream(const char *path, const char *mode=0);
    FileStream(FILE *f);
    FileStream(int fd);
    virtual ~FileStream();
    
    int fileno() const { return fd; }
    
    virtual bool isOpened() const;
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
  };
  
} // namespace april_io

#endif // STREAM_BUFFER_H
