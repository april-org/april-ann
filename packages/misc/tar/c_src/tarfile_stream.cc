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
#include "tarfile_stream.h"

namespace TAR {

  TARFileStream::TARFileStream(TARPackage *cpp_tar_package,
                               StreamInterface *tar_file,
                               size_t offset, size_t size) :
    cpp_tar_package(cpp_tar_package), tar_file(tar_file),
    offset(offset), size(size), pos(0) {
    cpp_tar_package->incOpenFilesCounter();
  }
  
  TARFileStream::~TARFileStream() {
    close();
  }
  
  bool TARFileStream::isOpened() const {
    return !tar_file.empty();
  }

  void TARFileStream::close() {
    // TODO: check errors
    if (!tar_file.empty()) {
      tar_file.reset();
      cpp_tar_package->decOpenFilesCounter();
    }
  }
  
  void TARFileStream::flush() {
    // nothing to do
  }
  
  int TARFileStream::setvbuf(int mode, size_t size) {
    // nothing to do
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
  
  bool TARFileStream::hasError() const {
    // TODO: check errors
    return false;
  }
  
  const char *TARFileStream::getErrorMsg() const {
    return StreamInterface::NO_ERROR_STRING;
  }
    
  bool TARFileStream::privateEof() const {
    return static_cast<size_t>(pos) >= size;
  }

  size_t TARFileStream::privateWrite(const char *buf, size_t size) {
    UNUSED_VARIABLE(buf);
    UNUSED_VARIABLE(size);
    ERROR_EXIT(128, "Read only stream\n");
    return 0;
  }
  
  size_t TARFileStream::privateRead(char *buf, size_t max_size) {
    if (pos >= size) return 0;
    tar_file->seek(SEEK_SET, offset + pos);
    max_size = AprilUtils::min(max_size, size - pos);
    size_t nbytes = tar_file->get(buf, max_size);
    pos += nbytes;
    return nbytes;
  }
  
  off_t TARFileStream::privateSeek(int whence, long offset) {
    off_t final;
    if (whence == SEEK_CUR && offset == 0) {
      return this->pos;
    }
    else {
      switch(whence) {
      case SEEK_CUR:
        final = tar_file->seek(SEEK_SET, this->offset + this->pos + offset) - this->offset;
        break;
      case SEEK_SET:
        final = tar_file->seek(SEEK_SET, this->offset + offset) - this->offset;
        break;
      case SEEK_END:
        final = tar_file->seek(SEEK_SET, this->offset + this->size - 1 + offset) - this->offset;
        break;
      default:
        final=0; // avoid compiler warnings
        ERROR_EXIT(128, "Unknown whence value\n");
      }
      if (final < 0) final = 0;
      else if (static_cast<size_t>(final) > this->size) final = this->size;
      this->pos = final;
      return final;
    }
  }
 
} // namespace TAR
