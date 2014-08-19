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

#include "handled_stream.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "buffered_stream.h"
#include "unused_variable.h"

namespace AprilIO {
  
  /**
   * @brief Specialization of BufferedStream and HandledStreamInterface for
   * input/output operations into a disk file.
   */
  class FileStream : public BufferedStream, public HandledStreamInterface {
    int fd,    ///< File descriptor.
      errnum,  ///< Last error number.
      flags;   ///< Flags given at open function.
    bool is_eof; ///< Indicator of EOF.
    
    /// Auxiliary method which allow to process returned values from C POSIX
    /// functions.
    template<typename T>
    T checkReturnValue(T ret_value);

  protected:
    virtual ssize_t fillBuffer(char *dest, size_t max_size);
    virtual ssize_t flushBuffer(const char *source, size_t max_size);
    virtual void closeStream();
    virtual off_t seekStream(int whence, int offset);
    virtual bool eofStream() const;

  public:
    
    /// Constructor from a path and a mode, in the same way as @c fopen.
    FileStream(const char *path, const char *mode=0);
    /// Constructor from a C @c FILE struct.
    FileStream(FILE *f);
    /// Constructor from a C file descriptor.
    FileStream(int fd);
    /// Destructor
    virtual ~FileStream();
    
    int fileno() const { return fd; }
    
    virtual bool isOpened() const;
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
  };
  
} // namespace AprilIO

#endif // STREAM_BUFFER_H
