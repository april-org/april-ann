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
#ifndef BUFFERED_ZIPFILE_H
#define BUFFERED_ZIPFILE_H

#include <zip.h>
#include <cstring>
#include "buffered_memory.h"
#include "error_print.h"
#include "unused_variable.h"
#include "zipio.h"

namespace ZIPIO {
  class ZIPFileStream : public april_io::BufferedInputStream {
  public:
    
    ZIPFileStream(ZIPPackage *zip_package, const char );
    /*
      ZIPFileStream(FILE *file);
      ZIPFileStream(int fd);
    */
    virtual ~ZIPFileStream();
  
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
}
#endif // BUFFERED_ZIPFILE_H
