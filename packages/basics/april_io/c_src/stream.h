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

#include <cstdio>
#include <cstring>

#include "constString.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "unused_variable.h"

#define READ_ONLY_STREAM(name)                                  \
  virtual size_t put(StreamInterface *source, size_t size) {    \
    UNUSED_VARIABLE(source);                                    \
    UNUSED_VARIABLE(size);                                      \
    ERROR_EXIT(128, "Read only " #name "\n");                   \
    return 0;                                                   \
  }                                                             \
  virtual size_t put(const char *source, size_t size) {         \
    UNUSED_VARIABLE(source);                                    \
    UNUSED_VARIABLE(size);                                      \
    ERROR_EXIT(128, "Read only " #name "\n");                   \
    return 0;                                                   \
  }                                                             \
  virtual int printf(const char *format, ...) {                 \
    UNUSED_VARIABLE(format);                                    \
    ERROR_EXIT(128, "Read only " #name "\n");                   \
    return 0;                                                   \
  }

#define WRITE_ONLY_STREAM(name)                                         \
  virtual size_t get(StreamInterface *dest, const char *delim) {        \
    UNUSED_VARIABLE(dest);                                              \
    UNUSED_VARIABLE(delim);                                             \
    ERROR_EXIT(128, "Write only " #name "\n");                          \
    return 0;                                                           \
  }                                                                     \
  virtual size_t get(StreamInterface *dest, size_t max_size, const char *delim) { \
    UNUSED_VARIABLE(dest);                                              \
    UNUSED_VARIABLE(max_size);                                          \
    UNUSED_VARIABLE(delim);                                             \
    ERROR_EXIT(128, "Write only " #name "\n");                          \
    return 0;                                                           \
  }                                                                     \
  virtual size_t get(char *dest, size_t max_size, const char *delim) {  \
    UNUSED_VARIABLE(dest);                                              \
    UNUSED_VARIABLE(max_size);                                          \
    UNUSED_VARIABLE(delim);                                             \
    ERROR_EXIT(128, "Write only " #name "\n");                          \
    return 0;                                                           \
  }

namespace april_io {
  /**
   * The StreamInterface is the basic public interface needed to implement a
   * Stream.
   */
  class StreamInterface : public Referenced {
  public:
    
    StreamInterface() : Referenced() { }
    
    virtual ~StreamInterface() { }
    
    /// Returns if the stream is properly opened and not EOF and no errors.
    virtual bool good() const = 0;

    /// Reads a string delimited by any of the given chars and puts it into the
    /// given Stream. If delim==0 then this method ends when dest->eof() is
    /// true.
    virtual size_t get(StreamInterface *dest, const char *delim = 0) = 0;
    
    /// Reads a string with max_size length and delimited by any of the given
    /// chars and puts it into the given Stream.
    virtual size_t get(StreamInterface *dest, size_t max_size,
                       const char *delim = 0) = 0;
    
    /// Reads a string of a maximum given size delimited by any of the given
    /// chars and puts it into the given char buffer.
    virtual size_t get(char *dest, size_t max_size, const char *delim = 0) = 0;
    
    /// Puts a string of a maximum given size taken from the given Stream.
    virtual size_t put(StreamInterface *source, size_t size) = 0;
    
    /// Puts a string of a maximum given size taken from the given char buffer.
    virtual size_t put(const char *source, size_t size) = 0;

    /// Puts a zero ended string, uses strlen to compute source length.
    virtual size_t put(const char *source) {
      return put(source, strlen(source));
    }
    
    /// Writes a set of values following the given format. Equals to C printf.    
    virtual int printf(const char *format, ...) = 0;
    
    /// Indicates if end-of-file flag is true.
    virtual bool eof() const = 0;
    
    ///////// ABSTRACT INTERFACE /////////
    
    /// Indicates if the stream is correctly open.
    virtual bool isOpened() const = 0;

    /// Closes the stream.
    virtual void close() = 0;
    
    /// Moves the stream cursor to the given offset from given whence position.
    virtual off_t seek(int whence = SEEK_CUR, int offset = 0) = 0;
    
    /// Forces to write pending data at stream object.
    virtual void flush() = 0;
    
    /// Modifies the behavior of the buffer.
    virtual int setvbuf(int mode, size_t size) = 0;
    
    /// Indicates if an error has been produced.
    virtual bool hasError() const = 0;
    
    /// Returns an internal string with the last error message.
    virtual const char *getErrorMsg() const = 0;
    
  protected:
    static char DUMMY_CHAR;
    static const char *NO_ERROR_STRING;
  };
  
  
  /**
   * The Stream is the parent class which needs to be dervied by I/O
   * facilities based in input/output buffers.
   */
  class StreamBuffer : public StreamInterface {
    const char *in_buffer;
    size_t in_buffer_pos, in_buffer_len;
    char *out_buffer;
    size_t out_buffer_pos, out_buffer_len;
    
  public:
    StreamBuffer();
    virtual ~StreamBuffer();
    
    virtual bool good() const;
    virtual size_t get(StreamInterface *dest, const char *delim = 0);
    virtual size_t get(StreamInterface *dest, size_t max_size, const char *delim = 0);
    virtual size_t get(char *dest, size_t max_size, const char *delim = 0);
    virtual size_t put(StreamInterface *source, size_t size);
    virtual size_t put(const char *source, size_t size);
    virtual int printf(const char *format, ...);
    virtual bool eof() const;
    
    ///////// ABSTRACT INTERFACE /////////
    
    // virtual bool isOpened() const = 0;
    // virtual void close() = 0;
    // virtual off_t seek(int whence, int offset) = 0;
    // virtual void flush() = 0;
    // virtual int setvbuf(int mode, size_t size) = 0;
    // virtual bool hasError() const = 0;
    // virtual const char *getErrorMsg() const = 0;
    
  protected:
    
    // Auxiliary protected methods

    virtual void resetBuffers();
    virtual size_t getInBufferPos() const;
    virtual size_t getOutBufferPos() const;
    virtual void resetOutBuffer();
    
    /// Indicates the available data in the current input buffer.
    virtual size_t getInBufferAvailableSize() const;
    
    /// Indicates the available data in the current output buffer.
    virtual size_t getOutBufferAvailableSize() const;
    
    /**
     * Moves the input buffer pointer a given number of bytes, when empty
     * buffer, this call will refill the buffer with new data from the real
     * stream.
     *
     * @param len - Size of the movement (<= available size).
     */
    virtual void moveInBuffer(size_t len);

    /**
     * Moves the output buffer pointer a given number of bytes, when full
     * buffer, this call will flush the data into the real stream.
     *
     * @param len - Size of the movement (<= available size).
     */
    virtual void moveOutBuffer(size_t len);
    
    ///// ABSTRACT METHODS /////
    
    virtual const char *nextInBuffer(size_t &buf_len) = 0;

    virtual char *nextOutBuffer(size_t &buf_len) = 0;

    virtual bool eofStream() const = 0;
    
    ////////////////////////////

  private:
    void trimInBuffer(const char *delim);
    
    /**
     * Returns a pointer to the current input buffer with at most max_size
     * bytes delimited by the given string of delimitiers.
     *
     * @param[out] buffer_len - Size of the returned pointer (<= max_size).
     * @param max_size - The maximum size required for the buffer.
     * @param delim - A string with delimitiers.
     *
     * @return A pointer to input buffer position.
     */
    virtual const char *getInBuffer(size_t &buffer_len, size_t max_size,
                                    const char *delim);
    
    /**
     * Returns a pointer to the current output buffer with at most max_size.
     *
     * @param[out] buffer_len - Size of the returned pointer (<= max_size).
     * @param max_size - The maximum size required for the buffer.
     *
     * @return A pointer to input buffer position.
     */
    virtual char *getOutBuffer(size_t &buffer_len, size_t max_size);

  };
  
} // namespace april_io

#endif // STREAM_H
