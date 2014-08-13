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

namespace gzio {

  class GZFileStream : public april_io::StreamInterface {
  public:
    
    /// Indicates if the extesion of a given filename corresponds with a
    /// gzipped file.
    static bool isGZ(const char *filename) {
      int len = strlen(filename);
      return (len > 3 && filename[len-3] == '.' &&
              filename[len-2] == 'g' &&
              filename[len-1] == 'z');
    }
  
    GZFileStream(const char *path, const char *mode);
    /*
      GZFileStream(FILE *file);
      GZFileStream(int fd);
    */
    virtual ~GZFileStream();
  
    // int fileno() const { return fd; }

    virtual bool good() const;
    virtual size_t get(StreamInterface *dest, const char *delim = 0);
    virtual size_t get(StreamInterface *dest, size_t max_size, const char *delim = 0);
    virtual size_t get(char *dest, size_t max_size, const char *delim = 0);
    virtual size_t put(StreamInterface *source, size_t size);
    virtual size_t put(const char *source, size_t size);
    virtual int printf(const char *format, ...);
    virtual bool eof() const;
    virtual bool isOpened() const ;
    virtual void close();
    virtual off_t seek(int whence=SEEK_CUR, int offset=0);
    virtual void flush();
    virtual int setvbuf(int mode, size_t size);
    virtual bool hasError() const;
    virtual const char *getErrorMsg() const;
    
  private:
    
    static const size_t DEFAULT_BUFFER_SIZE = 64*1024; // 64K
    
    gzFile f;
    char *in_buffer;
    size_t in_buffer_pos, in_buffer_len, max_buffer_size;
    bool write_flag;

    void prepareInBufferData();    
    void trimInBuffer(const char *delim);
    template<typename T>
    size_t templatizedGet(T &putOp, size_t max_size, const char *delim);
  };

} // namespace gzio

#endif // GZFILE_STREAM_H

