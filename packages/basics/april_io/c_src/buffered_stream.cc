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
#include <sys/types.h>
#include <unistd.h>

#include "maxmin.h"
#include "buffered_stream.h"

namespace april_io {
  
  BufferedStream::BufferedStream(size_t buf_size) : max_buffer_len(buf_size) {
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

  off_t BufferedStream::seek(int whence, int offset) {
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
}
