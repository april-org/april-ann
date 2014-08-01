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
#include "cfile_stream.h"

namespace april_io {
        
  CFileStream::CFileStream() : Stream(), f(0), need_close(false) {
  }
        
  CFileStream::CFileStream(FILE *f) : Stream(), f(f), need_close(false) {
  }
        
  CFileStream::CFileStream(const char *path, const char *mode) :
    Stream(), f(0), need_close(true) {
    f = fopen(path, mode);
    need_close = true;
  }
        
  CFileStream::~CFileStream() {
    close();
  }
        
  void CFileStream::close() {
    if (need_close && f != 0) fclose(f);
    f = 0;
  }
        
  void CFileStream::flush() {
    fflush(f);
  }
        
  bool CFileStream::isOpened() const {
    return f != 0;
  }
        
  bool CFileStream::eof() {
    return feof(f) != 0;
  }
        
  int CFileStream::seek(long offset, int whence) {
    return fseek(f, offset, whence);
  }
        
  size_t CFileStream::read(void *ptr, size_t size, size_t nmemb) {
    return fread(ptr, size, nmemb, f);
  }
        
  size_t CFileStream::write(const void *ptr, size_t size, size_t nmemb) {
    return fwrite(ptr, size, nmemb, f);
  }

} // namespace april_io
