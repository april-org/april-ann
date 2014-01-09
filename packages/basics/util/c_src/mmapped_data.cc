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
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>           // mmap() is defined in this header
#include <cstring>
#include "mmapped_data.h"
#include "error_print.h"

extern const char *__COMMIT_NUMBER__;

namespace april_utils {
  
  MMappedDataReader::MMappedDataReader(const char *path,
				       bool write,
				       bool shared) {
    if ((fd = open(path, O_RDWR)) < 0)
      ERROR_EXIT1(128,"Unable to open file %s\n", path);
    // find size of input file
    struct stat statbuf;
    if (fstat(fd, &statbuf) < 0) {
      ERROR_EXIT(128, "Error guessing filesize\n");
    }
    // mmap the input file
    mmapped_data_size = statbuf.st_size;
    int prot = PROT_READ;
    if (write) prot = prot | PROT_WRITE;
    int flags;
    if (shared) flags = MAP_SHARED;
    else flags = MAP_PRIVATE;
    if ((mmapped_data = static_cast<char*>(mmap(0, mmapped_data_size,
						prot, flags,
						fd, 0)))  == (caddr_t)-1)
      ERROR_EXIT(128, "mmap error\n");
    int magic = *(reinterpret_cast<int*>(mmapped_data));
    if (magic != MAGIC_NUMBER) ERROR_EXIT(128, "Incorrect endianism\n");
    pos = sizeof(int);
    commit_number = *(this->get<int>());
  }
  
  MMappedDataReader::~MMappedDataReader() {
    if (fd != -1) close(fd);
    munmap(mmapped_data, mmapped_data_size);
  }
  
  size_t MMappedDataReader::size() const { return mmapped_data_size - pos; }
  
  ////////////////////////////////////////////////////////////////
  
  MMappedDataWriter::MMappedDataWriter(const char *path) {
    if ((fd = open(path, O_CREAT | O_WRONLY | O_TRUNC,
		   S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH)) < 0)
      ERROR_EXIT1(128,"Unable to open file %s\n", path);
    int magic = MAGIC_NUMBER;
    this->put(&magic);
    int commit_number = atoi(__COMMIT_NUMBER__);
    this->put(&commit_number);
  }
  
  MMappedDataWriter::~MMappedDataWriter() {
    close(fd);
  }
}
