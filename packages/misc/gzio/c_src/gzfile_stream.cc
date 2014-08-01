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
#include "error_print.h"
#include "gzfile_stream.h"

namespace gzio {

  GZFileStream::GZFileStream() : Stream(), f(0), need_close(false) {
  }
        
  GZFileStream::GZFileStream(gzFile f) : Stream(), f(f), need_close(false) {
  }
        
  GZFileStream::GZFileStream(const char *path, const char *mode) :
    Stream(), f(0), need_close(true) {
    f = gzopen(path, mode);
    need_close = true;
  }
        
  GZFileStream::~GZFileStream() {
    close();
  }
  
  void GZFileStream::close() {
    if (need_close && f != 0) gzclose(f);
    f = 0;
  }
        
  void GZFileStream::flush() {
    gzflush(f, Z_SYNC_FLUSH);
  }
        
  bool GZFileStream::isOpened() const {
    return f != 0;
  }
        
  bool GZFileStream::eof() {
    return gzeof(f) != 0;
  }
        
  int GZFileStream::seek(long offset, int whence) {
    if (whence == SEEK_END) {
      ERROR_EXIT(256, "gzFile doesn't support whence=SEEK_END\n");
    }
    return gzseek(f, offset, whence);
  }
        
  size_t GZFileStream::read(void *ptr, size_t size, size_t nmemb) {
    return gzread(f, ptr, size*nmemb);
  }
        
  size_t GZFileStream::write(const void *ptr, size_t size, size_t nmemb) {
    return gzwrite(f, ptr, size*nmemb);
  }

} // namespace gzio
