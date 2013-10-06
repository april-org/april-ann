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

#include "referenced.h"
#include "april_assert.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>           // mmap() is defined in this header
#include "error_print.h"
#include "ignore_result.h"

namespace april_utils {

  const int MAGIC_NUMBER = 0x3333;

  class MMappedDataReader : public Referenced {
    char   *mmapped_data;
    size_t  mmapped_data_size;
    size_t  pos;
    int     fd;
  public:
    MMappedDataReader(const char *path) {
      if ((fd = open(path, O_RDWR)) < 0)
	ERROR_EXIT1(128,"Unable to open file %s\n", path);
      // find size of input file
      struct stat statbuf;
      if (fstat(fd, &statbuf) < 0) {
	ERROR_EXIT(128, "Error guessing filesize\n");
      }
      // mmap the input file
      mmapped_data_size = statbuf.st_size;
      if ((mmapped_data = static_cast<char*>(mmap(0, mmapped_data_size,
						  PROT_READ | PROT_WRITE,
						  MAP_SHARED,
						  fd, 0)))  == (caddr_t)-1)
	ERROR_EXIT(128, "mmap error\n");
      int magic = *(reinterpret_cast<int*>(mmapped_data));
      if (magic != MAGIC_NUMBER) ERROR_EXIT(128, "Incorrect endianism\n");
      pos = sizeof(int);
    }
  
    ~MMappedDataReader() {
      if (fd != -1) close(fd);
      munmap(mmapped_data, mmapped_data_size);
    }
  
    size_t size() const { return mmapped_data_size - pos; }
  
    template<typename T> T *get(size_t n=1) {
      size_t sz = sizeof(T)*n;
      if (sz + pos > mmapped_data_size)
	ERROR_EXIT(128, "Overflow reading from mmap\n");
      T *ptr = reinterpret_cast<T*>(mmapped_data+pos);
      pos   += sizeof(T)*n;
      if (pos == mmapped_data_size) { close(fd); fd=-1; }
      return ptr;
    }
  };

  class MMappedDataWriter : public Referenced {
    int fd;
  public:
    MMappedDataWriter(const char *path) {
      if ((fd = open(path, O_CREAT | O_WRONLY | O_TRUNC,
		     S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH)) < 0)
	ERROR_EXIT1(128,"Unable to open file %s\n", path);
      int magic = MAGIC_NUMBER;
      IGNORE_RESULT(write(fd, &magic,        sizeof(int)));
    }
    ~MMappedDataWriter() {
      close(fd);
    }
    template<typename T>
    void put(T *data, size_t n=1) {
      IGNORE_RESULT(write(fd, data, sizeof(T)*n));
    }
  };
}

#endif // MMAPPED_DATA_H
