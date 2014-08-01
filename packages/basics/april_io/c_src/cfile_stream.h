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
#ifndef CFILE_H
#define CFILE_H

#include <cstdlib>
#include <cstdio>
#include "stream.h"

namespace april_io {

  class CFileStream : public Stream {
    FILE *f;
    bool need_close;
  public:
    CFileStream();
    CFileStream(int fd);
    CFileStream(FILE *f);
    CFileStream(const char *path, const char *mode);
    virtual ~CFileStream();
    virtual void close();
    virtual void flush();
    virtual bool isOpened() const;
    virtual bool eof();
    virtual int seek(long offset, int whence);
    virtual size_t read(void *ptr, size_t size, size_t nmemb);
    virtual size_t write(const void *ptr, size_t size, size_t nmemb);
  };
  
} // namespace april_io

#endif // CFILE_H
