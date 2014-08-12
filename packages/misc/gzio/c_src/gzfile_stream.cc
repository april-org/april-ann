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
#include "error_print.h"
#include "gzfile_stream.h"
#include "maxmin.h"

using april_io::StreamInterface;

namespace gzio {

  GZFileStream::GZFileStream(const char *path, const char *mode) :
    StreamInterface(), f(0) {
    f = gzopen(path, mode);
    in_buffer       = new char[DEFAULT_BUFFER_SIZE+1];
    max_buffer_size = DEFAULT_BUFFER_SIZE;
    in_buffer_pos   = in_buffer_len = 0;
    write_flag      = (mode[0] == 'w' || mode[1] == '+');
  }

  /*
    GZFileStream::GZFileStream(FILE *file) : BufferedStream() {
    char mode[3];
    f = gzdopen(dup(fileno(file)), mode);
    }
    
    GZFileStream::GZFileStream(int fd) : BufferedStream() {
    char mode[3];
    f = gzdopen(dup(fd), mode);
    }
  */
          
  GZFileStream::~GZFileStream() {
    close();
  }
  
  void GZFileStream::prepareInBufferData() {
    // read only when the buffer is empty
    if (in_buffer_pos == in_buffer_len) {
      int nbytes = gzread(f, in_buffer, max_buffer_size);
      // TODO: check errors
      in_buffer_len = static_cast<size_t>(nbytes);
      in_buffer_pos = 0;
      // WARNING: strpbrk needs this '\0' at buffer end position
      in_buffer[in_buffer_len] = '\0';
    }
  }
  
  void GZFileStream::trimInBuffer(const char *delim) {
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
  size_t GZFileStream::templatizedGet(T &putOp, size_t max_size,
                                      const char *delim) {
    size_t dest_len=0;
    trimInBuffer(delim);
    // Read data until complete the given max_size or a delimitier is found, and
    // no errors are found.
    while( this->good() &&
           !putOp.hasError() &&
           dest_len < max_size ) {
      prepareInBufferData();
      // Condition of empty buffer, no data is available.
      if (in_buffer_pos >= in_buffer_len) break;
      size_t available_size = in_buffer_len - in_buffer_pos;
      size_t buf_len = april_utils::min(available_size, max_size - dest_len);
      if (delim != 0) {
        // WARNING: in_buffer needs a '\0' to indicate its size
        char *delim_pos = strpbrk(in_buffer + in_buffer_pos, delim);
        if (delim_pos != NULL) {
          ptrdiff_t ptr_diff = delim_pos - (in_buffer + in_buffer_pos);
          size_t diff = static_cast<size_t>(ptr_diff);
          buf_len = april_utils::min(buf_len, diff);
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

  bool GZFileStream::good() const {
    return isOpened() && !eof();
  }

  size_t GZFileStream::get(StreamInterface *dest, const char *delim) {
    return get(dest, SIZE_MAX, delim);
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
  size_t GZFileStream::get(StreamInterface *dest, size_t max_size,
                           const char *delim) {
    putOperatorStream put_op(dest);
    return templatizedGet(put_op, max_size, delim);
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
  size_t GZFileStream::get(char *dest, size_t max_size, const char *delim) {
    putOperatorBuffer put_op(dest);
    return templatizedGet(put_op, max_size, delim);
  }
  
  size_t GZFileStream::put(StreamInterface *source, size_t size) {
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

  size_t GZFileStream::put(const char *source, size_t size) {
    // TODO: check error conditions
    gzwrite(f, source, size);
    return size;
  }
  
  int GZFileStream::printf(const char *format, ...) {
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
  
  bool GZFileStream::eof() const {
    return (in_buffer_pos == in_buffer_len) && gzeof(f);
  }
  
  bool GZFileStream::isOpened() const {
    return f != NULL;
  }
  
  void GZFileStream::close() {
    gzclose(f);
    f = NULL;
  }
  
  off_t GZFileStream::seek(int whence, int offset) {
    if (whence == SEEK_CUR) {
      off_t real_pos    = gzseek(f, 0, SEEK_CUR);
      off_t current_pos = real_pos - (in_buffer_len - in_buffer_pos);
      if (offset == 0) return current_pos;
      offset -= (real_pos - current_pos);
    }
    in_buffer_pos = in_buffer_len = 0;
    return gzseek(f, offset, whence);
  }
  
  void GZFileStream::flush() {
    gzflush(f, Z_SYNC_FLUSH);
  }
  
  int GZFileStream::setvbuf(int mode, size_t size) {
    // TODO:
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
  
  bool GZFileStream::hasError() const {
    // TODO:
    return false;
  }
  
  const char *GZFileStream::getErrorMsg() const {
    // TODO:
    return StreamInterface::NO_ERROR_STRING;
  }
  
} // namespace gzio
