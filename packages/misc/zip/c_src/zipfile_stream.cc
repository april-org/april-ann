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
#include "zipfile_stream.h"

namespace ZIP {

  ZIPFileStream::ZIPFileStream(ZIPPackage *cpp_zip_package,
                               zip_file *file,
                               size_t size) :
    cpp_zip_package(cpp_zip_package), file(file),
    size(size), pos(0) {
  }
  
  ZIPFileStream::~ZIPFileStream() {
    close();
  }
  
  bool ZIPFileStream::isOpened() const {
    return file != 0;
  }

  void ZIPFileStream::close() {
    // TODO: check errors
    if (file != 0) zip_fclose(file);
    file = 0;
  }
  
  void ZIPFileStream::flush() {
    // nothing to do
  }
  
  int ZIPFileStream::setvbuf(int mode, size_t size) {
    // nothing to do
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
  
  bool ZIPFileStream::hasError() const {
    // TODO: check errors
    return false;
  }
  
  const char *ZIPFileStream::getErrorMsg() const {
    return StreamInterface::NO_ERROR_STRING;
  }
    
  bool ZIPFileStream::privateEof() const {
    return static_cast<size_t>(pos) >= size;
  }

  size_t ZIPFileStream::privateWrite(const char *buf, size_t size) {
    UNUSED_VARIABLE(buf);
    UNUSED_VARIABLE(size);
    ERROR_EXIT(128, "Read only stream\n");
    return 0;
  }
  
  size_t ZIPFileStream::privateRead(char *buf, size_t max_size) {
    int nbytes = zip_fread(file, buf, max_size);
    if (nbytes < 0) {
      // TODO: check errors
    }
    else {
      pos += nbytes;
    }
    return static_cast<size_t>(nbytes);
  }
  
  off_t ZIPFileStream::privateSeek(int whence, int offset) {
    if (whence == SEEK_CUR && offset == 0) {
      return pos;
    }
    else {
      ERROR_EXIT(128, "Unable to seek\n");
    }
    return 0;
  }
 
} // namespace ZIP
