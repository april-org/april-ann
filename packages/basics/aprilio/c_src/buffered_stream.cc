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
#include <cstdarg>
#include <cstddef>
#include <cstdlib>
extern "C" {
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
}

#include "maxmin.h"
#include "buffered_stream.h"

namespace AprilIO {
  
  BufferedStream::BufferedStream(size_t buf_size) :
    StreamBuffer(),
    max_buffer_len(buf_size) {
    in_buffer = new char[max_buffer_len];
    out_buffer = new char[max_buffer_len];
  }
  
  BufferedStream::~BufferedStream() {
    delete[] in_buffer;
    delete[] out_buffer;
  }
  
  void BufferedStream::close() {
    if (isOpened()) {
      flush();
      closeStream();
    }
  }

  off_t BufferedStream::seek(int whence, long offset) {
    if (whence == SEEK_CUR) {
      off_t real_pos    = seekStream(SEEK_CUR, 0);
      off_t current_pos = real_pos - getInBufferAvailableSize();
      if (offset == 0) return current_pos;
      offset -= (real_pos - current_pos);
    }
    flush();
    resetBuffers();
    return seekStream(whence, offset);
  }
  
  void BufferedStream::flush() {
    size_t nbytes = flushBuffer(out_buffer, getOutBufferPos());
    if (nbytes < max_buffer_len - getOutBufferPos()) {
      memmove(out_buffer, out_buffer + nbytes, max_buffer_len - nbytes);
    }
    resetOutBuffer();
  }
  
  int BufferedStream::setvbuf(int mode, size_t size) {
    // TODO:
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
    
  const char *BufferedStream::nextInBuffer(size_t &buf_len) {
    buf_len = fillBuffer(in_buffer, max_buffer_len);
    return in_buffer;
  }

  char *BufferedStream::nextOutBuffer(size_t &buf_len) {
    buf_len = max_buffer_len;
    return out_buffer;
  }

  ////////////////////////////////////////////////////////////
  
  BufferedInputStream::BufferedInputStream(size_t buf_size) :
    StreamInterface(), max_buffer_size(buf_size) {
    in_buffer       = new char[max_buffer_size + 1];
    in_buffer_pos   = in_buffer_len = 0;
  }
  
  BufferedInputStream::~BufferedInputStream() {
    delete[] in_buffer;
  }
  
  void BufferedInputStream::prepareInBufferData() {
    // read only when the buffer is empty
    if (in_buffer_pos == in_buffer_len) {
      int nbytes = privateRead(in_buffer, max_buffer_size);
      // TODO: check errors
      in_buffer_len = static_cast<size_t>(nbytes);
      in_buffer_pos = 0;
      // WARNING: strpbrk needs this '\0' at buffer end position
      in_buffer[in_buffer_len] = '\0';
    }
  }

  void BufferedInputStream::trimInBuffer(const char *delim) {
    if (delim != 0) {
      do {
        prepareInBufferData();
        // move buffer position as many delimitiers are in the buffer
        while(in_buffer_pos < in_buffer_len &&
              strchr(delim, in_buffer[in_buffer_pos]) != 0) {
          ++in_buffer_pos;
        }
        // continue reading data while all the buffer are delimitiers
      } while(in_buffer_pos == in_buffer_len && in_buffer_len > 0);
    }
  }

  // putOp is a predicate which wraps a StreamInterface or a char buffer.
  template<typename T>
  size_t BufferedInputStream::templatizedGet(T &putOp, size_t max_size,
                                             const char *delim,
                                             bool keep_delim) {
    size_t dest_len=0;
    if (!keep_delim) trimInBuffer(delim);
    // Read data until complete the given max_size or a delimitier is found, and
    // no errors are found.
    while( this->good() &&
           !putOp.hasError() &&
           dest_len < max_size ) {
      prepareInBufferData();
      // Condition of empty buffer, no data is available.
      if (in_buffer_pos >= in_buffer_len) break;
      size_t available_size = in_buffer_len - in_buffer_pos;
      size_t buf_len = AprilUtils::min(available_size, max_size - dest_len);
      if (delim != 0) {
        // WARNING: in_buffer needs a '\0' to indicate its size
        char *delim_pos = strpbrk(in_buffer + in_buffer_pos, delim);
        if (delim_pos != NULL) {
          if (delim_pos < in_buffer + in_buffer_len && keep_delim) ++delim_pos;
          ptrdiff_t ptr_diff = delim_pos - (in_buffer + in_buffer_pos);
          size_t diff = static_cast<size_t>(ptr_diff);
          buf_len = AprilUtils::min(buf_len, diff);
        }
      }
      // Put data into the given operator, which wraps a StreamInterface or a
      // char buffer.
      size_t len = putOp(in_buffer + in_buffer_pos, buf_len);
      in_buffer_pos += len;
      dest_len += len;
      // Delim true condition.
      if (len != available_size) break;
    }
    return dest_len;
  }

  bool BufferedInputStream::good() const {
    return isOpened() && !eof();
  }

  size_t BufferedInputStream::get(StreamInterface *dest, const char *delim,
                                  bool keep_delim) {
    return get(dest, SIZE_MAX, delim, keep_delim);
  }
  
  // Operator for put data into a StreamInterface object.
  struct putOperatorStream {
    StreamInterface *dest;
    putOperatorStream(StreamInterface *dest) : dest(dest) { }
    size_t operator()(char *buffer, size_t len) {
      return dest->put(buffer, len);
    }
    bool hasError() const { return dest->hasError(); }
  };
  // Get method.
  size_t BufferedInputStream::get(StreamInterface *dest, size_t max_size,
                                  const char *delim,
                                  bool keep_delim) {
    putOperatorStream put_op(dest);
    return templatizedGet(put_op, max_size, delim, keep_delim);
  }

  // Operator for put data into a char buffer.
  struct putOperatorBuffer {
    char *dest;
    putOperatorBuffer(char *dest) : dest(dest) { }
    size_t operator()(char *buffer, size_t len) {
      memcpy(dest, buffer, len);
      return len;
    }
    bool hasError() const { return false; }
  };
  // Get method.
  size_t BufferedInputStream::get(char *dest, size_t max_size,
                                  const char *delim, bool keep_delim) {
    putOperatorBuffer put_op(dest);
    return templatizedGet(put_op, max_size, delim, keep_delim);
  }
  
  size_t BufferedInputStream::put(StreamInterface *source, size_t size) {
    char *buf = new char[size];
    size_t source_len = 0;
    while( !this->hasError() &&
           source->good() &&
           source_len < size ) {
      size_t len = source->get(buf + source_len, size - source_len, 0);
      source_len += len;
    }
    size_t ret_value = put(buf, source_len);
    delete[] buf;
    return ret_value;
  }

  size_t BufferedInputStream::put(const char *source) {
    return put(source, strlen(source));
  }

  size_t BufferedInputStream::put(const char *source, size_t size) {
    // TODO: control errors
    return privateWrite(source, size);
  }
  
  int BufferedInputStream::printf(const char *format, ...) {
    va_list arg;
    char *aux_buffer;
    size_t len;
    va_start(arg, format);
    if (vasprintf(&aux_buffer, format, arg) < 0) {
      ERROR_EXIT(256, "Problem creating auxiliary buffer\n");
    }
    len = strlen(aux_buffer);
    if (len > 0) len = put(aux_buffer, len);
    free(aux_buffer);
    return len;
  }

  bool BufferedInputStream::eof() const {
    return (in_buffer_pos == in_buffer_len) && privateEof();
  }

  off_t BufferedInputStream::seek(int whence, long offset) {
    if (whence == SEEK_CUR) {
      off_t real_pos    = privateSeek(SEEK_CUR, 0);
      off_t current_pos = real_pos - (in_buffer_len - in_buffer_pos);
      if (offset == 0) return current_pos;
      offset -= (real_pos - current_pos);
    }
    in_buffer_pos = in_buffer_len = 0;
    return privateSeek(whence, offset);
  }

}
