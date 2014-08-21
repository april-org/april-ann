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
#ifndef MMAPPED_DATA_H
#define MMAPPED_DATA_H

extern "C" {
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>           // mmap() is defined in this header
}
#include "referenced.h"
#include "april_assert.h"
#include "error_print.h"
#include "ignore_result.h"

namespace april_utils {

  class MMappedDataReader : public Referenced {
    int commit_number;
    char   *mmapped_data;
    size_t  mmapped_data_size;
    size_t  pos;
    int     fd;
  public:
    MMappedDataReader(const char *path, bool write=true, bool shared=true);
    ~MMappedDataReader();
    size_t size() const;
    
    template<typename T> T *get(size_t n=1) {
      size_t sz = sizeof(T)*n;
      if (sz + pos > mmapped_data_size)
	ERROR_EXIT(128, "Overflow reading from mmap\n");
      T *ptr = reinterpret_cast<T*>(mmapped_data+pos);
      pos   += sizeof(T)*n;
      if (pos == mmapped_data_size) { close(fd); fd=-1; }
      return ptr;
    }
    int getCommitNumber() const { return commit_number; }
  };

  class MMappedDataWriter : public Referenced {
    int fd;
  public:
    MMappedDataWriter(const char *path);
    ~MMappedDataWriter();
    template<typename T>
    void put(const T *data, size_t n=1) {
      IGNORE_RESULT(write(fd, data, sizeof(T)*n));
    }
  };
}

#endif // MMAPPED_DATA_H
