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
extern "C" {
#include <stdint.h> // for SIZE_MAX (using stdint.h because cstdint needs c++11
                    // support)
}

#include "stream.h"
#include "stream_memory.h"
#include "unused_variable.h"

namespace april_io {
  
  Stream::Stream() : Referenced() {
  }

  Stream::~Stream() {
  }

  void Stream::trimInBuffer(const char *delim) {
    size_t pos, buf_len;
    do {
      const char *buf = getInBuffer(buf_len, SIZE_MAX, delim);
      pos = 0;
      while(pos < buf_len && strchr(delim, buf[pos])) ++pos;
      if (pos != buf_len) break;
      moveInBuffer(pos);
    } while(true); // the end condition is determined by break if above
  }
  
  bool Stream::good() const {
    return isOpened() && !eof() && !hasError();
  }
  
  size_t Stream::get(Stream *dest, const char *delim) {
    return get(dest, SIZE_MAX, delim);
  }

  size_t Stream::get(Stream *dest, size_t max_size, const char *delim) {
    const char *buf;
    size_t buf_len, dest_len=0;
    trimInBuffer(delim);
    while( !this->hasError() &&
           !this->eof() &&
           !dest->hasError() &&
           !dest->eof() &&
           dest_len < max_size &&
           (buf = getInBuffer(buf_len, max_size - dest_len, delim)) ) {
      size_t in_buffer_tail_size = getInBufferAvailableSize();
      size_t len = dest->put(buf, buf_len);
      moveInBuffer(len);
      dest_len += len;
      // delim true condition
      if (len != in_buffer_tail_size) break;
    }
    return dest_len;
  }
  
  size_t Stream::get(char *dest, size_t size, const char *delim) {
    const char *buf;
    size_t buf_len, dest_len=0;
    trimInBuffer(delim);
    while( dest_len < size &&
           !this->hasError() &&
           !this->eof() &&
           (buf = getInBuffer(buf_len, size - dest_len, delim)) ) {
      size_t in_buffer_tail_size = getInBufferAvailableSize();
      memcpy(dest + dest_len, buf, buf_len);
      moveInBuffer(buf_len);
      dest_len += buf_len;
      // delim true condition
      if (buf_len != in_buffer_tail_size) break;
    }
    return dest_len;
  }

  size_t Stream::put(Stream *source, size_t size) {
    char *buf;
    size_t buf_len, source_len = 0;
    while( !this->hasError() &&
           !source->hasError() &&
           source_len < size &&
           (buf = getOutBuffer(buf_len, size - source_len)) ) {
      size_t len = source->get(buf, buf_len, 0);
      moveOutBuffer(len);
      source_len += len;
    }
    return source_len;
  }

  size_t Stream::put(const char *source, size_t size) {
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

  int Stream::printf(const char *format, ...) {
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
} // namespace april_io
