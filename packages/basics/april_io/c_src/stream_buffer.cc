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
#include "maxmin.h"
#include "stream_buffer.h"

namespace april_io {
  const char *StreamBuffer::getInBuffer(size_t &buffer_len, size_t max_size,
                                        const char *delim) {
    buffer_len = april_utils::min(in_buffer_len - in_buffer_pos, max_size);
    if (delim != 0) {
      size_t pos = in_buffer_pos;
      while(pos < in_buffer_len && !strchr(delim, in_buffer[pos])) ++pos;
      buffer_len = pos - in_buffer_pos;
    }
    return in_buffer + in_buffer_pos;
  }
  
  size_t StreamBuffer::getInBufferAvailableSize() const {
    return in_buffer_len - in_buffer_pos;
  }

  char *StreamBuffer::getOutBuffer(size_t &buffer_len, size_t max_size) {
    buffer_len = april_utils::min(out_buffer_len - out_buffer_pos, max_size);
    return out_buffer + out_buffer_pos;
  }
  
  void StreamBuffer::moveInBuffer(size_t len) {
    if (len > in_buffer_len - in_buffer_pos) {
      ERROR_EXIT(128, "Read buffer overflow!!!\n");
    }
    in_buffer_pos += len;
    if (in_buffer_pos == in_buffer_len) {
      in_buffer_len = fillBuffer(in_buffer, max_buffer_len);
      in_buffer_pos = 0;
    }
  }
  
  void StreamBuffer::moveOutBuffer(size_t len) {
    if (len > out_buffer_len - out_buffer_pos) {
      ERROR_EXIT(128, "Write buffer overflow!!!\n");
    }
    out_buffer_pos += len;
    if (out_buffer_pos == out_buffer_len) {
      size_t nbytes = flushBuffer(out_buffer, out_buffer_len);
      if (nbytes < out_buffer_len) {
        memmove(out_buffer, out_buffer + nbytes, out_buffer_len - nbytes);
      }
      out_buffer_pos = 0;
    }
  }
  
  void StreamBuffer::resetBuffers() {
    in_buffer_pos  = in_buffer_len;
    out_buffer_pos = out_buffer_len;
  }
  
  StreamBuffer::StreamBuffer(size_t buf_size) : max_buffer_len(buf_size),
                                                in_buffer_pos(0),
                                                out_buffer_pos(0),
                                                in_buffer_len(0),
                                                out_buffer_len(buf_size) {
    in_buffer = new char[max_buffer_len];
    out_buffer = new char[max_buffer_len];
  }
  
  StreamBuffer::~StreamBuffer() {
    close();
    delete[] in_buffer;
    delete[] out_buffer;
  }
  
  void StreamBuffer::close() {
    if (isOpened()) {
      flush();
      closeStream();
    }
  }

  off_t StreamBuffer::seek(int whence, int offset) {
    flush();
    resetBuffers();
    int ret_value = seekStream(whence, offset);
    fillBuffer(in_buffer, max_buffer_len);
    return ret_value;
  }
  
  void StreamBuffer::flush() {
    flushBuffer(out_buffer, out_buffer_pos);
  }
  
  int StreamBuffer::setvbuf(int mode, size_t size) {
    // TODO:
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
  }
  
  bool StreamBuffer::eof() const {
    return (in_buffer_pos == in_buffer_len) && eofStream();
  }
}
