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
#ifndef BUFFERED_FILE_H
#define BUFFERED_FILE_H

#include <cstdlib>
#include "error_print.h"
#include "buffered_memory.h"
#include "referenced.h"

class FileWrapper {
  FILE *f;
  bool need_close;
public:
  FileWrapper(FileWrapper &other) : f(other.f), need_close(false) { }
  FileWrapper(FILE *f) : f(f), need_close(false) { }
  FileWrapper() : f(0), need_close(true) { }
  ~FileWrapper() {
    closeS();
  }
  bool isOpened() const { return f != 0; }
  bool openS(const char *path, const char *mode) {
    f = fopen(path, mode);
    need_close = true;
    return f != 0;
  }
  void closeS() {
    if (need_close && f != 0) fclose(f);
    f = 0;
  }
  size_t readS(void *ptr, size_t size, size_t nmemb) {
    return fread(ptr, size, nmemb, f);
  }
  size_t writeS(const void *ptr, size_t size, size_t nmemb) {
    return fwrite(ptr, size, nmemb, f);
  }
  int seekS(long offset, int whence) {
    return fseek(f, offset, whence);
  }
  void flushS() {
    fflush(f);
  }
  int printfS(const char *format, va_list &arg) {
    return vfprintf(f, format, arg);
  }
  bool eofS() const {
    return feof(f) != 0;
  }
};

typedef BufferedMemory<FileWrapper> BufferedFile;

#endif // BUFFERED_FILE_H