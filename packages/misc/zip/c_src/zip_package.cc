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
extern "C" {
#include <fcntl.h>
#include <unistd.h>
}
#include <cerrno>

#include "april_assert.h"
#include "zipfile_stream.h"
#include "zip_package.h"

using AprilIO::HandledStreamInterface;
using AprilIO::StreamInterface;
using april_utils::constString;

namespace ZIP {

  const size_t ZIPPackage::ERROR_BUFFER_SIZE = 1024;
  
  ZIPPackage::ZIPPackage(const char *path, const char *mode) :
    ArchivePackage() {
    constString mode_cstr(mode);
    int flags;
    init();
    if (mode_cstr.empty() || mode_cstr == "r") {
      flags = 0;
    }
    else if (mode_cstr == "r+") {
      flags = 0;
    }
    else if (mode_cstr == "w") {
      flags = ZIP_CREATE; // | ZIP_TRUNCATE;
    }
    else if (mode_cstr == "w+") {
      flags = ZIP_CREATE; // | ZIP_TRUNCATE;
    }
    else {
      flags = 0;
      ERROR_EXIT1(128, "Unknown given mode string '%s'\n", mode);
    }
    zip_package = zip_open(path, flags | ZIP_CHECKCONS, &zerr);
    serr = errno;
  }
  
  ZIPPackage::ZIPPackage(int fd) :
    ArchivePackage() {
    init();
    openFileDescriptor(fd);
  }
  
  ZIPPackage::ZIPPackage(HandledStreamInterface *handled_stream) :
    ArchivePackage() {
    init();
    openFileDescriptor(handled_stream->fileno());
  }
  
  ZIPPackage::ZIPPackage(StreamInterface *stream) :
    ArchivePackage() {
    init();
    HandledStreamInterface *handled_stream =
      dynamic_cast<HandledStreamInterface*>(stream);
    if (handled_stream == 0) zerr = ZIP_ER_OPEN;
    else openFileDescriptor(handled_stream->fileno());
  }
  
  ZIPPackage::~ZIPPackage() {
    close();
  }

  bool ZIPPackage::good() const {
    return zip_package != 0 && !hasError();
  }
  
  void ZIPPackage::close() {
    if (zip_package != 0) checkReturnedValue(zip_close(zip_package));
    zip_package = 0;
  }

  void ZIPPackage::init() {
    zerr = 0;
    serr = 0;
    zip_package = 0;
    error_buffer = new char[ERROR_BUFFER_SIZE+1];
  }

  size_t ZIPPackage::getNumberOfFiles() {
    april_assert(zip_package != 0);
    return static_cast<size_t>(zip_get_num_entries(zip_package,0));
  }

  const char *ZIPPackage::getNameOf(size_t idx) {
    struct zip_stat sb;
    ::zip_stat_index(zip_package, static_cast<int>(idx), 0, &sb);
    if (sb.valid & ZIP_STAT_NAME) {
      return sb.name;
    }
    else {
      return 0;
    }
  }
  
  void ZIPPackage::openFileDescriptor(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags & O_WRONLY || flags & O_RDWR) {
      zerr = ZIP_ER_INVAL;
    }
    else {
      int dup_fd = dup(fd);
      zip_package = zip_fdopen(dup_fd, ZIP_CHECKCONS, &zerr);
      serr = errno;
    }
  }
  
  bool ZIPPackage::hasError() const {
    return zerr != 0;
  }
  
  const char *ZIPPackage::getErrorMessage() {
    zip_error_to_str(error_buffer.get(), ERROR_BUFFER_SIZE, zerr, serr); 
    return error_buffer.get();
  }

  StreamInterface *ZIPPackage::openFile(const char *name, int flags) {
    april_assert(zip_package != 0);
    struct zip_stat sb;
    ::zip_stat(zip_package, name, 0, &sb);
    if (sb.valid & ZIP_STAT_SIZE) {
      size_t size = sb.size;
      zip_file *file = checkReturnedValue(zip_fopen(zip_package, name, flags));
      if (file == 0) return 0;
      return new ZIPFileStream(this, file, size);
    }
    else {
      return 0;
    }
  }
  
  StreamInterface *ZIPPackage::openFile(size_t idx, int flags) {
    april_assert(zip_package != 0);
    struct zip_stat sb;
    ::zip_stat_index(zip_package, static_cast<zip_uint64_t>(idx), 0, &sb);
    if (sb.valid & ZIP_STAT_SIZE) {
      size_t size = sb.size;
      zip_file *file =
        checkReturnedValue(zip_fopen_index(zip_package,
                                           static_cast<zip_uint64_t>(idx),
                                           flags));
      if (file == 0) return 0;
      return new ZIPFileStream(this, file, size);
    }
    else {
      return 0;
    }
  }
  
  template <typename T>
  T ZIPPackage::checkReturnedValue(T code) {
    april_assert(zip_package != 0);
    if (code != 0) {
      zip_error_get(zip_package, &zerr, &serr);
    }
    return code;
  }
  
} // namespace ZIP
