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

extern "C" {
#include <zip.h>
}

#include <cstring>
#include "buffered_stream.h"
#include "error_print.h"
#include "smart_ptr.h"
#include "unused_variable.h"
#include "zip_package.h"

namespace ZIP {
  class ZIPFileStream : public AprilIO::BufferedInputStream {
    friend class ZIPPackage;
  public:
    
    virtual ~ZIPFileStream();
  
    virtual bool isOpened() const;
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
    
    april_utils::SharedPtr<ZIPPackage> cpp_zip_package;
    zip_file *file;
    size_t size;
    off_t pos;

    ZIPFileStream(ZIPPackage *cpp_zip_package, zip_file *file, size_t size);
  };
}
#endif // BUFFERED_ZIPFILE_H
