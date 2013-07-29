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
#include <sys/mman.h>           // mmap() is defined in this header
#include <cstring>
#include <cstdio>
#include "matlab.h"
#include "endianism.h"
#include "ignore_result.h"
#include "error_print.h"

using april_utils::swap_bytes_in_place;

template<typename T>
uint32_t readMatrixData(MatrixFloat *m, FILE *f,
			const uint32_t nbytes, bool swap) {



  T v;
  MatrixFloat::col_major_iterator m_it(m->begin());
  uint32_t bytes_read=0;
  while(bytes_read < nbytes) {
    assert(m_it != m->end());
    size_t n;
    do {
      n = fread(&v, sizeof(T), 1, f);
    } while(n == 0);
    if (n < 0) ERROR_EXIT(128, "Unexpected EOF\n");
    bytes_read += n*sizeof(T);
    if (swap) swap_bytes_in_place(v);
    *m_it = static_cast<float>(v);
    ++m_it;
  }
  return bytes_read;
}

template<typename T>
bool readDataArray(FILE *f, T *data, size_t n, bool swap) {
  if (fread(data, sizeof(T), n, f) < n) return false;
  if (swap && sizeof(T)>1)
    for (size_t i=0; i<n; ++i) swap_bytes_in_place(data[i]);
  /*
    for (size_t i=0; i<n; ++i) printf("%u ", data[i]);
    printf("\n");
  */
  return true;
}

template<typename T>
int readTagAndData(FILE *f, T *data, size_t max, bool swap, size_t &N) {
  size_t bytes_read;
  char aux_tag[8];
  if (!readDataArray(f, aux_tag, 8, swap)) return false;
  bytes_read = 8;
  // FIXME: In the documentation of MAT FORMAT it says that [0] and [1] bytes
  // must be ZERO to not be compressed, but I found that must be [2] and [3]
  // bytes. Probably is an ENDIANISM problem...
  if (aux_tag[2] != 0 || aux_tag[3] != 0) {
    // COMPRESSED FORMAT
    uint16_t tag[2];
    memcpy(tag+NUMBER_OF_BYTES, aux_tag, sizeof(uint16_t));
    memcpy(tag+DATA_TYPE, aux_tag+sizeof(uint16_t), sizeof(uint16_t));
    if (swap) for (size_t i=0; i<2; ++i) swap_bytes_in_place(tag[i]);
    N = tag[NUMBER_OF_BYTES]/sizeof(T);
    if (N > max)
      ERROR_EXIT2(128, "Maximum overflow, found %d, expected %d\n", N, max);
    memcpy(data, aux_tag+2*sizeof(uint16_t), tag[NUMBER_OF_BYTES]);
  }
  else {
    // NON COMPRESSED FORMAT
    uint32_t tag[2];
    memcpy(&tag[DATA_TYPE], aux_tag, sizeof(uint32_t));
    memcpy(&tag[NUMBER_OF_BYTES], aux_tag+sizeof(uint32_t), sizeof(uint32_t));
    if (swap) for (size_t i=0; i<2; ++i) swap_bytes_in_place(tag[i]);
    N = tag[NUMBER_OF_BYTES]/sizeof(T);
    if (N > max)
      ERROR_EXIT2(128, "Maximum overflow, found %d, expected %d\n", N, max);
    if (!readDataArray(f, data, N, swap)) ERROR_EXIT(128, "Unexpected EOF\n");
    bytes_read += tag[NUMBER_OF_BYTES];
  }
  return bytes_read;
}

bool MatFileReader::readNextDataBlock() {
  if (current_data_tag_header[NUMBER_OF_BYTES] > 0) return true;
  if (!readDataArray(f, current_data_tag_header, 2, swap_bytes)) {
    fclose(f);
    f = 0;
    return false;
  }
  if (current_data_tag_header[DATA_TYPE] != MATRIX)
    ERROR_EXIT1(128, "Only MATRIX data type is allowed, found %d\n",
		current_data_tag_header[DATA_TYPE]);
  return true;
}

void MatFileReader::decreaseNumBytes(uint32_t n) {
  if (current_data_tag_header[NUMBER_OF_BYTES] < n)
    ERROR_EXIT(128, "Incorrect number of bytes\n");
  current_data_tag_header[NUMBER_OF_BYTES] -= n;
}

MatFileReader::MatFileReader(const char *path) {
  if ((fd = open(filename, O_RDONLY)) < 0)
    ERROR_EXIT1(128,"Unable to open  file %s\n", path);
  if ((mmapped_data = static_cast<char*>(mmap(0, filesize,
					      PROT_READ, MAP_SHARED,
					      fd, 0)))  == (caddr_t)-1)
    ERROR_EXIT(128, "mmap error\n");
  // EXTRACT HEADER
  header = getCurrent<HeaderType*>();
  addToCurrent(HEADER_SIZE);
  //
  union {
    char magic_char[2];
    uint16_t magic_int;
  } Endianism;
  Endianism.magic_int = header->magic;
  swap_bytes = (Endianism.magic_char[0]=='M' && Endianism.magic_char[1]=='I');
  if (swap_bytes)
    ERROR_EXIT1("ENDIANISM LOGIC NOT IMPLEMENTED\n");
  /*
    printf("%s\n%d\n%c%c %d\n", header_text, version,
    endianism.magic[0], endianism.magic[1], swap_bytes);
  */
  current_data_tag_header[NUMBER_OF_BYTES] = 0;
}

MatFileReader::~MatFileReader() {
  if (f != 0) fclose(f);
}

DataElement *MatFileReader::getNextDataElement() {
  DataElement *data;
  FullDataElement *full_data = getCurrent<FullDataElement*>();
  if (data->isSmall()) {
    // small data element
    SmallDataElement small_data = getCurrent<SmallDataElement*>();
    addToCurrent(SMALL_TAG_SIZE);
    small_data->data = getCurrent<void*>();
    addToCurrent(SMALL_DATA_SIZE);
    data = small_data;
  }
  else {
    addToCurrent(TAG_SIZE);
    full_data->data = getCurrent<void*>();
    addToCurrent(data->number_of_bytes);
    data = full_data;
  }
  return data;
}

MatrixFloat *MatFileReader::readNextMatrix(char *name, bool col_major) {
  size_t N;
  //
  if (!readNextDataBlock()) return 0;
  if (f == 0) return 0;
  
  //printf("FLAGS\n");
  // FLAGS
  char flags[8];
  decreaseNumBytes(readTagAndData(f, flags, 8, swap_bytes, N));
  // FIXME: In the documentation of MAT FORMAT it says that [3] of FLAG is the
  // class. I found it at the byte [0]. Probably another ENDIANISM problem...
  if (flags[0] != CL_SINGLE && flags[0] != CL_DOUBLE)
    ERROR_EXIT(256, "Not supported MATRIX type, only SINGLE and"
	       " DOUBLE are allowed\n");
  
  //printf("DIMS\n");
  // DIMENSIONS
  uint32_t num_dims;
  int32_t dims[MAX_NUM_DIMS];
  decreaseNumBytes(readTagAndData(f, dims, MAX_NUM_DIMS, swap_bytes, N));
  num_dims = N;
  //printf("NUM DIMS %d\n", num_dims);
  
  //printf("NAME\n");
  // NAME
  decreaseNumBytes(readTagAndData(f, name, MAX_NAME_SIZE, swap_bytes, N));
  name[N] = '\0';
  //printf("%s\n", name);
  
  // MATRIX DATA
  uint32_t matrix_data_flags[2];
  if (!readDataArray(f, matrix_data_flags, 2, swap_bytes))
    ERROR_EXIT(128, "Unexpected EOF\n");
  decreaseNumBytes(2*sizeof(uint32_t));
  
  MatrixFloat *m = new MatrixFloat(static_cast<int>(num_dims), dims,
				   (col_major)?CblasColMajor:CblasRowMajor);
  if (matrix_data_flags[DATA_TYPE] == SINGLE)
    decreaseNumBytes(readMatrixData<float>(m, f,
					   matrix_data_flags[NUMBER_OF_BYTES],
					   swap_bytes));
  else if (matrix_data_flags[DATA_TYPE] == DOUBLE)
    decreaseNumBytes(readMatrixData<double>(m, f,
					    matrix_data_flags[NUMBER_OF_BYTES],
					    swap_bytes));
  if (getNumBytes() > 0) {
    ERROR_PRINT1("Implemented MAT format reader is not complete,"
		 "%d bytes remain to be read\n", getNumBytes());
    char *aux = new char[getNumBytes()];
    fread(aux, sizeof(char), getNumBytes(), f);
    decreaseNumBytes(getNumBytes());
  }
  return m;
}
