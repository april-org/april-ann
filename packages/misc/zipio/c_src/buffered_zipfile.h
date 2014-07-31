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
#ifndef BUFFERED_ZIPFILE_H
#define BUFFERED_ZIPFILE_H

#include <zip.h>
#include <cstring>
#include "buffered_memory.h"
#include "error_print.h"
#include "unused_variable.h"

/// Opens the file at zip index 0 or use the given opened zip_file
class ZIPFileWrapper {
  zip *z;
  zip_file *f;
  bool need_close;
  size_t file_size, read_size;
public:
  static bool isZIP(const char *filename) {
    int len = strlen(filename);
    return (len > 4 && filename[len-4] == '.' &&
	    filename[len-3] == 'z' &&
	    filename[len-2] == 'i' &&
	    filename[len-1] == 'p');
  }
  
  ZIPFileWrapper(ZIPFileWrapper &other) : f(other.f), need_close(false) { }
  ZIPFileWrapper(zip *z, zip_file *f) : z(z), f(f), need_close(false) { }
  ZIPFileWrapper() : f(0), need_close(true) { }
  ~ZIPFileWrapper() {
    closeS();
  }
  bool isOpened() const { return f != 0; }
  bool openS(const char *path, const char *mode) {
    int err, flags=0;
    if (mode != 0) {
      if (mode[0] == 'w') {
        // flags = flags | ZIP_CREATE; // | ZIP_TRUNCATE;
        ERROR_EXIT(128, "zip files writing not implemented\n");
      }
    }
    z = zip_open(path, 0, &err);
    f = zip_fopen_index(z, 0, 0);
    struct zip_stat stat;
    zip_stat_index(z, 0, 0, &stat);
    file_size = stat.size;
    read_size = 0;
    need_close = true;
    return f != 0;
  }
  void closeS() {
    if (need_close && z != 0) {
      if (f != 0) zip_fclose(f);
      zip_close(z);
    }
    f = 0;
  }
  size_t readS(void *ptr, size_t size, size_t nmemb) {
    size_t n = zip_fread(f, ptr, size*nmemb);
    read_size += n;
    return n;
  }
  size_t writeS(const void *ptr, size_t size, size_t nmemb) {
    UNUSED_VARIABLE(ptr);
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(nmemb);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return 0;
  }
  int seekS(long offset, int whence) {
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(whence);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return 0;
  }
  void flushS() {
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
  }
  int printfS(const char *format, va_list &arg) {
    UNUSED_VARIABLE(format);
    UNUSED_VARIABLE(arg);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return 0;
  }
  bool eofS() const {
    return read_size == file_size;
  }
};

typedef BufferedMemory<ZIPFileWrapper> BufferedZIPFile;

#endif // BUFFERED_GZFILE_H
