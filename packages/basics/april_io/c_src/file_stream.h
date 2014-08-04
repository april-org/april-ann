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

#include "constString.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "stream_buffer.h"
#include "unused_variable.h"

namespace april_io {
  class FileStream : public StreamBuffer {
    /// File descriptor.
    int fd, errnum;
    bool is_eof;
    
    template<typename T>
    bool checkReturnValue(T ret_value);

  protected:
    virtual ssize_t fillBuffer(char *dest, size_t max_size);
    virtual ssize_t flushBuffer(const char *source, size_t max_size);
    virtual void closeStream() = 0;
    virtual off_t seekStream(int whence, int offset) = 0;
    virtual bool eofStream() const;

  public:
    
    FileStream(const char *path, const char *mode);
    FileStream(int fd);
    virtual ~FileStream();
    
    /// Calls isOpened method of stream property.
    virtual bool isOpened() const;
    
    /// Calls close method of stream property.
    virtual void close();
    
    /// Indicates if an error has been produced.
    virtual bool hasError() const;
    
    /// Returns an internal string with the last error message.
    virtual const char *getErrorMsg() const;
  };
  
} // namespace april_io

#endif // STREAM_BUFFER_H
