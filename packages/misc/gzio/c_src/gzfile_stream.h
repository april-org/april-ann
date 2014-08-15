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
#ifndef GZFILE_STREAM_H
#define GZFILE_STREAM_H

#include <zlib.h>
#include <cstring>

#include "buffered_stream.h"

namespace GZIO {

  class GZFileStream : public april_io::BufferedInputStream {
  public:
    
    GZFileStream(const char *path, const char *mode);
    /*
      GZFileStream(FILE *file);
      GZFileStream(int fd);
    */
    virtual ~GZFileStream();
  
    // int fileno() const { return fd; }

    virtual bool isOpened() const ;
    virtual void close();
    virtual void flush();
    virtual int setvbuf(int mode, size_t size);
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
    
  protected:

    virtual bool privateEof() const;    
    virtual size_t privateWrite(const char *buf, size_t size);
    virtual size_t privateRead(char *buf, size_t max_size);
    virtual off_t privateSeek(int whence, int offset);
    
  private:
    
    gzFile f;
    bool write_flag;

  };

} // namespace GZIO

#endif // GZFILE_STREAM_H

