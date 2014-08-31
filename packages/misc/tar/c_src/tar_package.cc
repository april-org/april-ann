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
#include "file_stream.h"
#include "tarfile_stream.h"
#include "tar_package.h"
#include "unused_variable.h"

using AprilIO::FileStream;
using AprilIO::HandledStreamInterface;
using AprilIO::StreamInterface;
using AprilUtils::constString;
using AprilUtils::string;

namespace TAR {

  TARPackage::TARPackage(const char *path, const char *mode) :
    ArchivePackage() {
    constString mode_cstr(mode);
    if (mode_cstr != "r") {
      ERROR_EXIT(128, "TARPackage only can open files for reading");
    }
    tar_file = new FileStream(path, mode);
    init();
  }
  
  TARPackage::TARPackage(int fd) :
    ArchivePackage() {
    openFileDescriptor(fd);
    init();
  }
  
  TARPackage::TARPackage(StreamInterface *stream, bool reopen) :
    ArchivePackage() {
    if (reopen) {
      HandledStreamInterface *handled_stream =
        dynamic_cast<HandledStreamInterface*>(stream);
      if (handled_stream == 0) {
        ERROR_EXIT(128, "Impossible to reopen the given stream\n");
      }
      else {
        openFileDescriptor(handled_stream->fileno());
      }
    }
    else {
      tar_file = stream;
    }
    init();
  }
  
  TARPackage::~TARPackage() {
    close();
  }

  bool TARPackage::good() const {
    return !tar_file.empty() && !hasError();
  }
  
  void TARPackage::close() {
    is_closed = true;
    tryClose();
  }

  void TARPackage::init() {
    size_t BLOCK_SIZE = 512;
    char block[BLOCK_SIZE+1];
    num_open_files = 0;
    is_closed = false;
    // reads the header from the TAR stream
    off_t p = tar_file->seek(SEEK_SET, 0);
    while(true) {
      size_t nbytes = tar_file->get(block, BLOCK_SIZE);
      constString block_str(block, BLOCK_SIZE);
      if (nbytes == 0 || !AprilUtils::strncchr(block, '\0', BLOCK_SIZE)) break;
      Header header;
      decode(block, BLOCK_SIZE, header);
      if (header.typeflag == 0) {
        FileInfo info(header.pathname, tar_file->seek(), header.size);
        name2index[string(header.pathname)] = list.size();
        list.push_back(info);
      }
      p = tar_file->seek(SEEK_CUR,
                         static_cast<size_t>(BLOCK_SIZE *
                                             ceilf(static_cast<double>(header.size)/BLOCK_SIZE)));
      if (p<0) {
        ERROR_EXIT1(128, "%s\n", tar_file->getErrorMsg());
      }
    }
  }
  
  size_t TARPackage::getNumberOfFiles() {
    april_assert(!tar_file.empty());
    return list.size();
  }

  const char *TARPackage::getNameOf(size_t idx) {
    if (idx >= getNumberOfFiles()) return 0;
    return list[idx].pathname;
  }
  
  void TARPackage::openFileDescriptor(int fd) {
    int flags = fcntl(fd, F_GETFL);
    if (flags & O_WRONLY || flags & O_RDWR) {
      ERROR_EXIT(128, "TARPackage is only for read-only\n");
    }
    else {
      tar_file = new FileStream(fd);
    }
  }
  
  bool TARPackage::hasError() const {
    return tar_file->hasError();
  }
  
  const char *TARPackage::getErrorMsg() {
    return tar_file->getErrorMsg();
  }

  StreamInterface *TARPackage::openFile(const char *name, int flags) {
    april_assert(!tar_file.empty());
    size_t *index = name2index.find(name);
    // TODO: make an error code
    if (index == 0) return 0;
    return openFile(*index, flags);
  }
  
  StreamInterface *TARPackage::openFile(size_t idx, int flags) {
    UNUSED_VARIABLE(flags);
    april_assert(!tar_file.empty());
    // TODO: make an error code
    if (idx >= getNumberOfFiles()) return 0;
    FileInfo &info = list[idx];
    TARFileStream *file = new TARFileStream(this, tar_file.get(),
                                            info.offset, info.size);
    return file;
  }
  
  void TARPackage::incOpenFilesCounter() {
    ++num_open_files;
  }
  
  void TARPackage::decOpenFilesCounter() {
    --num_open_files;
    if (is_closed) tryClose();
  }
  
  void TARPackage::tryClose() {
    if (num_open_files == 0 && !tar_file.empty()) {
      tar_file->close();
      tar_file.reset();
    }
  }
  
  static size_t octal(char *data, size_t length) {
    size_t len = strnlen(data, length);
    size_t base = 1;
    size_t num = 0;
    for (size_t i=len; i>0; --i) {
      num  = num + ( data[i-1] - 48 ) * base;
      base = base << 3; // multiply by 8
    }
    return num;
  }

  void TARPackage::decode(const char *block, size_t block_size,
                          Header &header) {
    // sanity check
    april_assert(sizeof(Header) - 9*sizeof(size_t) - sizeof(header.pathname) <= block_size);
    size_t pos=0;
    //
#define READ_HEADER_FIELD(field) do {                                   \
      april_assert(pos + sizeof(header.field) <= block_size);           \
      memcpy(header.field, block+pos, sizeof(header.field));            \
      pos+=sizeof(header.field);                                        \
    } while(false)
    //
    READ_HEADER_FIELD(name);
    READ_HEADER_FIELD(octal_mode);
    READ_HEADER_FIELD(octal_uid);
    READ_HEADER_FIELD(octal_gid);
    READ_HEADER_FIELD(octal_size);
    READ_HEADER_FIELD(octal_mtime);
    READ_HEADER_FIELD(octal_chksum);
    READ_HEADER_FIELD(octal_typeflag);
    READ_HEADER_FIELD(linkname);
    READ_HEADER_FIELD(magic);
    READ_HEADER_FIELD(version);
    READ_HEADER_FIELD(uname);
    READ_HEADER_FIELD(gname);
    READ_HEADER_FIELD(octal_devmajor);
    READ_HEADER_FIELD(octal_devminor);
    READ_HEADER_FIELD(prefix);
    //
#undef READ_HEADER_FIELD
    april_assert(pos <= block_size);
    // convert octal values
    //
#define OCTAL(field) \
    header.field = octal(header.octal_##field, sizeof(header.octal_##field))
    //
    OCTAL(mode);
    OCTAL(uid);
    OCTAL(gid);
    OCTAL(size);
    OCTAL(mtime);
    OCTAL(chksum);
    OCTAL(typeflag);
    OCTAL(devmajor);
    OCTAL(devminor);
    //
#undef OCTAL
    header.pathname[0] = '\0';
    if (header.prefix[0] != '\0') {
      strncpy(header.pathname, header.prefix, sizeof(header.prefix));
      header.pathname[sizeof(header.prefix)] = '\0';
      strncat(header.pathname, "/", 1u);
    }
    strncat(header.pathname, header.name, sizeof(header.name));
  }
  
} // namespace TAR
