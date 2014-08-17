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
#ifndef TAR_PACKAGE_H
#define TAR_PACKAGE_H

#include "archive_package.h"
#include "handled_stream.h"
#include "hash_table.h"
#include "mystring.h"
#include "referenced.h"
#include "smart_ptr.h"
#include "stream.h"
#include "vector.h"

namespace TAR {

  // forward declaration
  class TARFileStream;

  class TARPackage : public AprilIO::ArchivePackage {
    friend class TARFileStream;
  public:
  
    TARPackage(const char *path, const char *mode);
    
    TARPackage(int fd);
    
    TARPackage(AprilIO::StreamInterface *stream, bool reopen=false);
    
    virtual ~TARPackage();

    virtual bool good() const;
    
    virtual bool hasError() const;
  
    virtual const char *getErrorMsg();
  
    virtual void close();

    virtual size_t getNumberOfFiles();
    
    virtual const char *getNameOf(size_t idx);
    
    virtual AprilIO::StreamInterface *openFile(const char *name, int flags);

    virtual AprilIO::StreamInterface *openFile(size_t idx, int flags);
    
  private:
    
    /// The position and size of the file inside the package.
    struct FileInfo {
      april_utils::string pathname; ///< The filename.
      size_t offset; ///< Offset from package stream begin.
      size_t size;   ///< Size of the file.
      FileInfo() { }
      FileInfo(const char *name, size_t off, size_t s) :
        pathname(name), offset(off), size(s) { }
    };
    
    /// Header of each file inside the TAR package
    struct Header {
      char name[100];
      char octal_mode[8];     // octal
      char octal_uid[8];      // octal
      char octal_gid[8];      // octal
      char octal_size[12];    // octal
      char octal_mtime[12];   // octal
      char octal_chksum[8];   // octal
      char octal_typeflag[1]; // octal
      char linkname[100];
      char magic[6];
      char version[2];
      char uname[32];
      char gname[32];
      char octal_devmajor[8]; // octal
      char octal_devminor[8]; // octal
      char prefix[155];
      //
      size_t mode, uid, gid, size, mtime, chksum, typeflag, devmajor, devminor;
      // derived field
      char pathname[260];
    };
    
    typedef april_utils::hash<april_utils::string, size_t> Name2IndexHash;
    typedef april_utils::vector<FileInfo> FileInfoList;
    
    Name2IndexHash name2index;
    FileInfoList   list;
    april_utils::SharedPtr<AprilIO::StreamInterface> tar_file;
    int num_open_files;
    bool is_closed;
    
    void init();
    void openFileDescriptor(int fd);
    void incOpenFilesCounter();
    void decOpenFilesCounter();
    void tryClose();
    void decode(const char *block, size_t block_size, Header &header);
  };

} // namespace TAR

#endif // TAR_PACKAGE_H
