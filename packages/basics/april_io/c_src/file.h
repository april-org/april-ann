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
#ifndef FILE_H
#define FILE_H

#include <cstdio>
#include <cstring>

#include "constString.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "unused_variable.h"

namespace april_io {
  const size_t DEFAULT_BUFFER_LEN = 4096;
  
  /**
   * The File is a wrapper over a Stream and allows to perform complex
   * read/write calls.
   */
  class File {
  protected:
    /// The stream from where data is read
    Stream *stream;
    /// It measures the number of bytes written or read
    size_t total_bytes;
    /// Reading buffer, writings are done directly
    char *buffer;
    /// Reserved size of the buffer
    int max_buffer_len;
    /// Current position of first valid char
    int buffer_pos;
    /// Number of chars chars read
    int buffer_len;
  
    /// Moves the buffer to the left and reads new data to fill the right part
    bool moveAndFillBuffer();
    
    /// Increases the size of the buffer and reads new data to fill it
    bool resizeAndFillBuffer();
    
    /// Moves buffer_pos until the first valid char is not in delim string
    bool trim(const char *delim);
  
    /// Forces to read from the file in the next getToken
    void setBufferAsFull() {
      buffer_pos = max_buffer_len;
      buffer_len = max_buffer_len;
    }
    
  public:
    
    File(Stream *stream);
    virtual ~File();

    void setExpectedSize(size_t sz) { stream->setExpectedSize(sz); }
    
    /// Calls isOpened method of stream property.
    virtual bool isOpened() const;

    /// Returns if the stream is properly opened and not EOF.
    virtual bool good() const;
    
    /// Calls close method of stream property.
    virtual void close();
   
    // be careful, this method returns a pointer to internal buffer
    
    /// Returns a string of the given maximum size. Be careful, the string is
    /// pointing to the internal buffer, so if you need to perform a copy if you
    /// need it. Otherwise, the string may be destroyed with the next read.
    virtual constString getToken(int size);
    
    // be careful, this method returns a pointer to internal buffer
    
    /// Returns a string delimited by any char of the given string. Be careful,
    /// the string is pointing to the internal buffer, so if you need to perform
    /// a copy if you need it. Otherwise, the string may be destroyed with the
    /// next read.
    virtual constString getToken(const char *delim);
    
    // be careful, this method returns a pointer to internal buffer
    
    /// Returns a whole line of the file (a string delimited by \n). Be careful,
    /// the string is pointing to the internal buffer, so if you need to perform
    /// a copy if you need it. Otherwise, the string may be destroyed with the
    /// next read.
    virtual constString extract_line();
    
    // be careful, this method returns a pointer to internal buffer
    
    /// Returns a whole line of the file (a string delimited by \n), but avoiding
    /// lines which begins with #. Lines beginning with # are taken as commented
    /// lines. Be careful, the string is pointing to the internal buffer, so if
    /// you need to perform a copy if you need it. Otherwise, the string may be
    /// destroyed with the next read.
    virtual constString extract_u_line();
    
    /// Writes a buffer of chars, given its length, similar to fwrite.
    virtual size_t write(const void *buffer, size_t len);

    /// Writes a set of values following the given format. Equals to C printf.    
    virtual int printf(const char *format, ...);
    
    /// Moves the file cursor to the given offset from given whence position.
    virtual int seek(int whence, int offset);
    
    /// Forces to write pending data at stream object.
    virtual void flush();
    
    /// Returns the value of the counter of read/written bytes.
    virtual size_t getTotalBytes() const;
  };
  
} // namespace april_io

#endif // FILE_H
