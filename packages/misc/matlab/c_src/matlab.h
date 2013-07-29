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
#ifndef MATLAB_H
#define MATLAB_H

#include <cstdio>
#include <stdint.h>
#include "matrixFloat.h"

#define MAX_NAME_SIZE    256
#define HEADER_SIZE      128
#define TAG_SIZE         8
#define SMALL_TAG_SIZE   4
#define SMALL_DATA_SIZE  4
#define MAX_NUM_DIMS     32

#define DATA_TYPE 0
#define NUMBER_OF_BYTES 1

class MatFileReader {
  enum MatFileDataTypes {
    INT8=1, UINT8=2, INT16=3, UINT16=4, INT32=5, UINT32=6,
    SINGLE=7, RESERVED1=8, DOUBLE=9, RESERVED2=10, RESERVED3=11,
    INT64=12, UINT64=13, MATRIX=14,
    COMPRESSED=15, UTF8=16, UTF16=17, UTF32=18,
    NUMBER_OF_DATA_TYPES,
  };
  enum MatFileClasses {
    CL_CELL_ARRAY=1, CL_STRUCTURE=2, CL_OBJECT=3, CL_CHAR=4, CL_SPARSE=5,
    CL_DOUBLE=6, CL_SINGLE=7, CL_INT8, CL_UINT8,
    L_INT16, CL_UINT16, CL_INT32, CL_UINT32,
    NUMBER_OF_CLASSES,
  };

  struct HeaderType {
    const char text[HEADER_TEXT_SIZE];
    const uint16_t version;
    const uint16_t magic;
  };
  class DataElement {
  public:
    virtual ~DataElement() { }
    virtual uint32_t getDataType() = 0;
    virtual uint32_t getNumberOfBytes() = 0;
  };
  struct FullDataElement : public DataElement {
    union {
      const char hbytes[4];
      const uint32_t data_type;
    };
    union {
      const char lbytes[4];
      const uint32_t number_of_bytes;
    };
  public:
    const void *data;
    virtual uint32_t getDataType() { return data_type; }
    virtual uint32_t getNumberOfBytes() { return number_of_bytes }
    bool isSmall() const {
      return (data->hbytes[2] != 0 || data->hbytes[3] != 0);
    }
  };
  struct SmallDataElement : public DataElement {
    union {
      const char hbytes[2];
      const uint16_t data_type;
    };
    union {
      const char lbytes[2];
      const uint16_t number_of_bytes;
    };
  public:
    const void *data;
    virtual uint32_t getDataType() { return static_cast<uint32_t>(data_type); }
    virtual uint32_t getNumberOfBytes() { return static_cast<uint32_t>(number_of_bytes); }
  };
  
  size_t mmapped_data_pos;
  char *mmapped_data;
  int   fd;
  headerType *header;
  
  char header_text[HEADER_TEXT_SIZE+1];
  int16_t version;

  bool swap_bytes;
  uint32_t current_data_tag_header[2];
  bool readNextDataBlock();
  void decreaseNumBytes(uint32_t n);
  uint32_t getNumBytes() const { return current_data_tag_header[NUMBER_OF_BYTES]; }
  
  template<typename T>
  T getCurrent() { return static_cast<T>(mmapped_data + mmapped_data_pos); }
  void  addToCurrent(size_t p) { mmapped_data_pos += p; }
  
public:
  MatFileReader(const char *path);
  ~MatFileReader();
  MatrixFloat *readNextMatrix(char *name, bool col_major=false);
};

#endif // MATLAB_H
