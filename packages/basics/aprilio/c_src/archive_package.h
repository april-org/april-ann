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
  
  /**
   * @brief Class which represent interface of package formats as ZIP or TAR.
   */
  class ArchivePackage : public Referenced {
  public:
   
    /// Constructor.
    ArchivePackage() : Referenced() { }
    
    /// Destructor.
    virtual ~ArchivePackage() { }
    
    /// Indicates if the object is in good state.
    virtual bool good() const = 0;
    
    /// Indicates if a call has produced an error.
    virtual bool hasError() const = 0;
    
    /// Returns the error message from last call error.
    virtual const char *getErrorMsg() = 0;
    
    /// Closes the package.
    virtual void close() = 0;
    
    /// Returns the number of files contained inside the package.
    virtual size_t getNumberOfFiles() = 0;
    
    /// Returns the name of the file with index idx.
    virtual const char *getNameOf(size_t idx) = 0;
    
    /**
     * @brief Opens a file given its name and a flags integer.
     *
     * @note The flags will change depending in the package.
     *
     * @param name - The filename as it is indicated by getNameOf() method.
     *
     * @param flags - The flags to open the file, which indicates the package
     * library how to find the given filename.
     *
     * @return A StreamInterface pointer or NULL if the call fails.
     */
    virtual StreamInterface *openFile(const char *name, int flags) = 0;
    
    /**
     * @brief Opens a file given its index and a flags integer.
     *
     * @note The flags will change depending in the package.
     *
     * @param idx - The index of the file inside the package.
     *
     * @param flags - The flags to open the file, which indicates the package
     * library how to find the filename by using its index idx.
     *
     * @return A StreamInterface pointer or NULL if the call fails.
     */
    virtual StreamInterface *openFile(size_t idx, int flags) = 0;

  };
    
} // namespace AprilIO

#endif // ARCHIVE_PACKAGE_H
