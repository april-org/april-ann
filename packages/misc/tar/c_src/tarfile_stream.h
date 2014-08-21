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
#ifndef BUFFERED_TARFILE_H
#define BUFFERED_TARFILE_H

#include <cstring>
#include "buffered_stream.h"
#include "error_print.h"
#include "smart_ptr.h"
#include "unused_variable.h"
#include "tar_package.h"

namespace TAR {
  class TARFileStream : public AprilIO::BufferedInputStream {
    friend class TARPackage;
  public:
    
    virtual ~TARFileStream();
  
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
    
    april_utils::SharedPtr<TARPackage> cpp_tar_package;
    april_utils::SharedPtr<StreamInterface> tar_file;
    size_t offset, size, pos;
    
    TARFileStream(TARPackage *cpp_tar_package, StreamInterface *tar_file,
                  size_t offset, size_t size);
  };
}
#endif // BUFFERED_TARFILE_H
