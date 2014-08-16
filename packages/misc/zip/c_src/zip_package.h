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
#ifndef ZIP_PACKAGE_H
#define ZIP_PACKAGE_H

extern "C" {
#include <zip.h>
}

#include "archive_package.h"
#include "handled_stream.h"
#include "referenced.h"
#include "smart_ptr.h"
#include "stream.h"

namespace ZIP {

  // forward declaration
  class ZIPFileStream;

  class ZIPPackage : public AprilIO::ArchivePackage {
    friend class ZIPFileStream;
  public:
  
    ZIPPackage(const char *path, const char *mode);
    
    ZIPPackage(int fd);
    
    ZIPPackage(AprilIO::HandledStreamInterface *stream);
    
    ZIPPackage(AprilIO::StreamInterface *stream);
    
    virtual ~ZIPPackage();

    virtual bool good() const;
    
    virtual bool hasError() const;
  
    virtual const char *getErrorMessage();
  
    virtual void close();

    virtual size_t getNumberOfFiles();
    
    virtual const char *getNameOf(size_t idx);
    
    virtual AprilIO::StreamInterface *openFile(const char *name, int flags);

    virtual AprilIO::StreamInterface *openFile(size_t idx, int flags);
    
  private:
  
    zip *zip_package;
    int zerr, serr;
    static const size_t ERROR_BUFFER_SIZE;
    april_utils::UniquePtr<char> error_buffer;
    int num_open_files;
    bool is_closed;
    
    void init();
    void openFileDescriptor(int fd);

    template<typename T>
    T checkReturnedValue(T code);
    void incOpenFilesCounter();
    void decOpenFilesCounter();
    void tryClose();
  };

} // namespace ZIP

#endif // ZIP_PACKAGE_H
