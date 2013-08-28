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
#include "buffered_memory.h"

class FileWrapper {
  FILE *f;
public:
  FileWrapper() : f(0) { }
  ~FileWrapper() {
    if (f != 0) closeS();
  }
  bool openS(const char *path, const char *mode) {
    f = fopen(path, mode);
    return f != 0;
  }
  void closeS() {
    fclose(f);
    f = 0;
  }
  size_t readS(void *ptr, size_t size, size_t nmemb) {
    return fread(ptr, size, nmemb, f);
  }
  size_t writeS(void *ptr, size_t size, size_t nmemb) {
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
  bool eofS() {
    return feof(f) != 0;
  }
};

typedef BufferedMemory<FileWrapper> BufferedFile;

#endif // BUFFERED_FILE_H
