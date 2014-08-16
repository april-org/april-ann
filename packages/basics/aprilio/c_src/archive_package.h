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
#ifndef ARCHIVE_PACKAGE_H
#define ARCHIVE_PACKAGE_H

#include "mystring.h"
#include "referenced.h"
#include "smart_ptr.h"
#include "stream.h"

namespace AprilIO {
  
  class ArchivePackage : public Referenced {
  public:
    
    ArchivePackage() : Referenced() { }
    
    virtual ~ArchivePackage() { }
    
    virtual bool good() const = 0;
    
    virtual bool hasError() const = 0;
    
    virtual const char *getErrorMessage() = 0;
    
    virtual void close() = 0;

    virtual size_t getNumberOfFiles() = 0;
    
    virtual const char *getNameOf(size_t idx) = 0;
    
    virtual StreamInterface *openFile(const char *name, int flags) = 0;
    
    virtual StreamInterface *openFile(size_t idx, int flags) = 0;

  };
    
} // namespace AprilIO

#endif // ARCHIVE_PACKAGE_H
