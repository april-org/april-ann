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
#include <cstdlib>

#include "error_print.h"
#include "gzfile_stream.h"
#include "maxmin.h"

using AprilIO::StreamInterface;

namespace GZIO {

  GZFileStream::GZFileStream(const char *path, const char *mode) :
    BufferedInputStream(), f(0) {
    f = gzopen(path, mode);
    write_flag = (mode[0] == 'w' || mode[1] == '+');
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
  
  bool GZFileStream::isOpened() const {
    return f != NULL;
  }
  
  void GZFileStream::close() {
    gzclose(f);
    f = NULL;
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

  bool GZFileStream::privateEof() const {
    return gzeof(f);
  }

  size_t GZFileStream::privateRead(char *buf, size_t max_size) {
    // TODO: check errors
    int nbytes = gzread(f, buf, max_size);
    return static_cast<size_t>(nbytes);
  }
  
  size_t GZFileStream::privateWrite(const char *buf, size_t size) {
    // TODO: check error conditions
    gzwrite(f, buf, size);
    return size;
  }
  
  off_t GZFileStream::privateSeek(int whence, int offset) {
    return gzseek(f, offset, whence);
  }
  
} // namespace gzio
