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
#include <cstdlib>
#include "maxmin.h"
#include "mystring.h"
#include "stream.h"
// #include "stream_memory.h"
#include "unused_variable.h"

namespace AprilIO {

  char StreamInterface::DUMMY_CHAR = '\0';
  const char *StreamInterface::NO_ERROR_STRING = "Success";
  
  StreamBuffer::StreamBuffer() : StreamInterface(),
                                 in_buffer(0), in_buffer_pos(0),
                                 in_buffer_len(0),
                                 out_buffer(0), out_buffer_pos(0),
                                 out_buffer_len(0) {
  }
  
  StreamBuffer::~StreamBuffer() {
  }

  void StreamBuffer::trimInBuffer(const char *delim, size_t max) {
    if (delim != 0) {
      size_t pos, buf_len, total=0;
      // size_t delim_len = strlen(delim);
      do {
        const char *buf = getInBuffer(buf_len, SIZE_MAX, 0, false);
        // pos = AprilUtils::strnspn(buf, buf_len, delim, delim_len);
        pos = 0;
        while(pos < buf_len && strchr(delim, buf[pos]) != 0 && total < max) ++pos,++total;
        moveInBuffer(pos);
      } while(pos == buf_len && buf_len > 0 && total < max);
    }
  }
  
  bool StreamBuffer::good() const {
    return isOpened() && !eof() && !hasError();
  }
  
  size_t StreamBuffer::get(StreamInterface *dest, const char *delim,
                           bool keep_delim) {
    return get(dest, SIZE_MAX, delim, keep_delim);
  }

  size_t StreamBuffer::get(StreamInterface *dest, size_t max_size,
                           const char *delim, bool keep_delim) {
    const char *buf;
    size_t buf_len, dest_len=0;
    while( this->good() &&
           !dest->hasError() &&
           dest_len < max_size &&
           (buf = getInBuffer(buf_len, max_size - dest_len, delim,
                              keep_delim)) ) {
      size_t in_buffer_available_size = getInBufferAvailableSize();
      size_t len = dest->put(buf, buf_len);
      bool has_delim = keep_delim && len>0 && strchr(delim,buf[len-1]);
      moveInBuffer(len);
      dest_len += len;
      // delim true condition
      if (has_delim || len != in_buffer_available_size) break;
    }
    if (!keep_delim) trimInBuffer(delim, 1);
    return dest_len;
  }
  
  size_t StreamBuffer::get(char *dest, size_t max_size, const char *delim,
                           bool keep_delim) {
    const char *buf;
    size_t buf_len, dest_len=0;
    while( dest_len < max_size &&
           this->good() &&
           (buf = getInBuffer(buf_len, max_size - dest_len, delim,
                              keep_delim)) ) {
      size_t in_buffer_available_size = getInBufferAvailableSize();
      memcpy(dest + dest_len, buf, buf_len);
      bool has_delim = keep_delim && buf_len>0 && strchr(delim, buf[buf_len-1]);
      moveInBuffer(buf_len);
      dest_len += buf_len;
      // delim true condition
      if (has_delim || buf_len != in_buffer_available_size) break;
    }
    if (!keep_delim) trimInBuffer(delim, 1);
    return dest_len;
  }

  size_t StreamBuffer::put(StreamInterface *source, size_t size) {
    char *buf;
    size_t buf_len, source_len = 0;
    while( !this->hasError() &&
           source->good() &&
           source_len < size &&
           (buf = getOutBuffer(buf_len, size - source_len)) ) {
      size_t len = source->get(buf, buf_len, 0);
      moveOutBuffer(len);
      source_len += len;
    }
    return source_len;
  }

  size_t StreamBuffer::put(const char *source, size_t size) {
    char *buf;
    size_t buf_len, source_len = 0;
    while( source_len < size &&
           !this->hasError() &&
           (buf = getOutBuffer(buf_len, size - source_len)) ) {
      memcpy(buf, source + source_len, buf_len);
      moveOutBuffer(buf_len);
      source_len += buf_len;
    }
    return source_len;
  }

  size_t StreamBuffer::put(const char *source) {
    return put(source, strlen(source));
  }

  int StreamBuffer::printf(const char *format, ...) {
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

  bool StreamBuffer::eof() const {
    return (in_buffer_pos == in_buffer_len) && eofStream();
  }
  
  //////////////////////////////////////////////////////////////////////////

  void StreamBuffer::resetBuffers() {
    in_buffer_pos  = in_buffer_len  = 0;
    out_buffer_pos = out_buffer_len = 0;
  }
  
  void StreamBuffer::resetOutBuffer() {
    out_buffer_pos = out_buffer_len = 0;
  }

  size_t StreamBuffer::getInBufferPos() const {
    return in_buffer_pos;
  }
  
  size_t StreamBuffer::getOutBufferPos() const {
    return out_buffer_pos;
  }
  
  const char *StreamBuffer::getInBuffer(size_t &buffer_len, size_t max_size,
                                        const char *delim, bool keep_delim) {
    if (in_buffer == 0) in_buffer = nextInBuffer(in_buffer_len);
    buffer_len = AprilUtils::min(in_buffer_len - in_buffer_pos, max_size);
    if (delim != 0) {
      size_t pos  = in_buffer_pos;
      size_t last = pos + buffer_len;
      /*
        pos = in_buffer_pos + AprilUtils::strncspn(in_buffer + pos, buffer_len,
        delim, strlen(delim));
      */
      while(pos < last && strchr(delim, in_buffer[pos]) == 0) ++pos;
      if (pos < last && keep_delim) ++pos;
      buffer_len = pos - in_buffer_pos;
    }
    return in_buffer + in_buffer_pos;
  }
  
  char *StreamBuffer::getOutBuffer(size_t &buffer_len, size_t max_size) {
    if (out_buffer == 0) out_buffer = nextOutBuffer(out_buffer_len);
    buffer_len = AprilUtils::min(out_buffer_len - out_buffer_pos, max_size);
    return out_buffer + out_buffer_pos;
  }

  size_t StreamBuffer::getInBufferAvailableSize() const {
    return in_buffer_len - in_buffer_pos;
  }
  
  size_t StreamBuffer::getOutBufferAvailableSize() const {
    return out_buffer_len - out_buffer_pos;
  }
  
  void StreamBuffer::moveInBuffer(size_t len) {
    if (len > getInBufferAvailableSize()) {
      ERROR_EXIT(128, "Read buffer overflow!!!\n");
    }
    in_buffer_pos += len;
    if (in_buffer_pos == in_buffer_len) {
      in_buffer = nextInBuffer(in_buffer_len);
      in_buffer_pos = 0;
    }
  }
  
  void StreamBuffer::moveOutBuffer(size_t len) {
    if (len > getOutBufferAvailableSize()) {
      ERROR_EXIT(128, "Write buffer overflow!!!\n");
    }
    out_buffer_pos += len;
    if (out_buffer_pos == out_buffer_len) {
      flush();
      out_buffer = nextOutBuffer(out_buffer_len);
      out_buffer_pos = 0;
    }
  }
  
} // namespace AprilIO
