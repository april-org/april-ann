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
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "constString.h"
#include "file_stream.h"

namespace april_io {

  template<typename T>
  T FileStream::checkReturnValue(T ret_value) {
    errnum = (ret_value < 0) ? errno : 0;
    return ret_value;
  }

  ssize_t FileStream::fillBuffer(char *dest, size_t max_size) {
    ssize_t ret_value, total_size;
    do {
      size_t len = max_size - total_size;
      if ((ret_value = read(fd, dest, len)) > 0) {
        total_size += ret_value;
      }
      else if (ret_value == 0) is_eof = true;
    } while( checkReturnValue(ret_value)<0 &&
             errnum==EINTR &&
             total_size < max_size );
    return (errnum==0) ? total_size : ret_value;
  }

  ssize_t FileStream::flushBuffer(const char *source, size_t max_size) {
    ssize_t ret_value, total_size;
    do {
      if ((ret_value = write(fd, source, max_size)) > 0) total_size += ret_value;
    } while( checkReturnValue(ret_value)<0 && errnum==EINTR );
    return (errnum==0) ? total_size : ret_value;
  }

  void FileStream::closeStream() {
    if (checkReturnValue(::close(fd)) == 0) fd = -1;
  }
  off_t FileStream::seekStream(int whence, int offset) {
    return checkReturnValue(lseek(fd, whence, offset));
  }
  
  FileStream::FileStream(const char *path, const char *mode) :
    StreamBuffer() {
    constString mode_cstr(mode);
    int flags;
    if (mode_cstr == "r") {
      flags = O_RDONLY;
    }
    else if (mode_cstr == "r+") {
      flags = O_RDWR;
    }
    else if (mode_cstr == "w") {
      flags = O_WRONLY | O_CREAT | O_TRUNC;
    }
    else if (mode_cstr == "w+") {
      flags = O_RDWR | O_CREAT | O_TRUNC;
    }
    else if (mode_cstr == "a") {
      flags = O_WRONLY | O_APPEND;
    }
    else if (mode_cstr == "a+") {
      flags = O_RDWR | O_CREAT | O_APPEND;
    }
    fd = checkReturnValue(open(path, flags));
  }
  
  FileStream::FileStream(int fd) : StreamBuffer() {
    fd = checkReturnValue(dup(fd));
  }
  
  bool FileStream::isOpened() { return fd >= 0; }

  bool FileStream::eofStream() const {
    return is_eof;
  }

  bool FileStream::hasError() const {
    return errnum != 0;
  }
  
  const char *FileStream::getErrorMsg() const {
    return strerror(errnum);
  }
  
}
