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
#ifndef BUFFERED_GZFILE_H
#define BUFFERED_GZFILE_H

#include <zlib.h>
#include <cstring>
#include "buffered_memory.h"

class GZFileWrapper {
  gzFile f;
  bool need_close;
public:
  static bool isGZ(const char *filename) {
    int len = strlen(filename);
    return (len > 3 && filename[len-3] == '.' &&
	    filename[len-2] == 'g' &&
	    filename[len-1] == 'z');
  }
  
  GZFileWrapper(GZFileWrapper &other) : f(other.f), need_close(false) { }
  GZFileWrapper(gzFile f) : f(f), need_close(false) { }
  GZFileWrapper() : f(0), need_close(true) { }
  ~GZFileWrapper() {
    closeS();
  }
  bool isOpened() const { return f != 0; }
  bool openS(const char *path, const char *mode) {
    f = gzopen(path, mode);
    need_close = true;
    return f != 0;
  }
  void closeS() {
    if (need_close && f != 0) gzclose(f);
    f = 0;
  }
  size_t readS(void *ptr, size_t size, size_t nmemb) {
    return gzread(f, ptr, size*nmemb);
  }
  size_t writeS(const void *ptr, size_t size, size_t nmemb) {
    return gzwrite(f, ptr, size*nmemb);
  }
  int seekS(long offset, int whence) {
    if (whence == SEEK_END)
      ERROR_EXIT(256, "gzFile doesn't support whence=SEEK_END\n");
    return gzseek(f, offset, whence);
  }
  void flushS() {
    gzflush(f, Z_SYNC_FLUSH);
  }
  int printfS(const char *format, va_list &arg) {
    char *aux_buffer;
    size_t len;
    if (vasprintf(&aux_buffer, format, arg) < 0)
      ERROR_EXIT(256, "Problem creating auxiliary buffer\n");
    len = strlen(aux_buffer);
    if (len > 0) len = gzwrite(f, aux_buffer, len);
    free(aux_buffer);
    return len;
  }
  bool eofS() const {
    return gzeof(f) != 0;
  }
};

typedef BufferedMemory<GZFileWrapper> BufferedGZFile;

#endif // BUFFERED_GZFILE_H

