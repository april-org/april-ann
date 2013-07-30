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
#include "matrixString.h"
#include "list.h"

#define MAX_NAME_SIZE    256
#define HEADER_SIZE      128
#define HEADER_TEXT_SIZE 124
#define TAG_SIZE         8
#define SMALL_TAG_SIZE   4
#define SMALL_DATA_SIZE  4
#define MAX_NUM_DIMS     32

#define DATA_TYPE 0
#define NUMBER_OF_BYTES 1

class MatFileReader : public Referenced {
public:
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
    CL_INT16, CL_UINT16, CL_INT32, CL_UINT32,
    NUMBER_OF_CLASSES,
  };
private:
  struct HeaderType {
    char text[HEADER_TEXT_SIZE];
    uint16_t version;
    uint16_t magic;
  };
  
  struct FullDataTag {
    union {
      char hbytes[4];
      uint32_t data_type;
    };
    union {
      char lbytes[4];
      uint32_t number_of_bytes;
    };
  };

  struct SmallDataTag {
    union {
      char hbytes[2];
      uint16_t data_type;
    };
    union {
      char lbytes[2];
      uint16_t number_of_bytes;
    };
  };

  /************* DATA ELEMENTS ****************/
public:
  class CellArrayDataElement;

  class DataElement : public Referenced {
    friend class MatFileReader;
  protected:
    const char *data;
    size_t data_pos;
    DataElement(const char *ptr) :
    Referenced(), data(ptr), data_pos(0) { }
  public:
    DataElement() : data(0), data_pos(0) { }
    virtual ~DataElement() { }
    virtual uint32_t getDataType() const = 0;
    virtual uint32_t getNumberOfBytes() const = 0;
    virtual size_t   getSizeOf() const = 0;
    template<typename T>
    T getData() const { return reinterpret_cast<T>(data); }
    void reset() { data_pos = 0; }
    // SUB-ELEMENTS
    DataElement *getNextSubElement();
    // ONLY IF MATRIX ARRAY
    MatrixFloat *getMatrix(char *name, size_t maxsize, bool col_major=false);
    uint32_t getClass();
    // ONLY IF CELL ARRAY
    CellArrayDataElement *getCellArray(char *name, size_t maxsize);
    // ONLY IF CHAR ARRAY
    MatrixString *getMatrixString(char *name, size_t maxsize);
  };

  class CellArrayDataElement : public Referenced {
    int *dims, *stride, num_dims, total_size;
    DataElement **elements;
    int computeRawPos(const int *coords);
  public:
    CellArrayDataElement(const int *dims, int num_dims);
    ~CellArrayDataElement();
    DataElement *getElementAt(const int *coords, int n);
    DataElement *getElementAt(int raw_idx);
    void setElementAt(const int *coords, int n, DataElement *e);
    void setElementAt(int raw_idx, DataElement *e);
    int getNumDim() const { return num_dims; }
    int getDimSize(int i) const { return dims[i]; }
    const int *getDimPtr() const { return dims; }
    int getSize() const { return total_size; }
  };
  
private:
  struct FullDataElement : public DataElement {
    const FullDataTag *tag;
  public:
    FullDataElement(const char *ptr) :
      DataElement(ptr+TAG_SIZE),
    tag(reinterpret_cast<const FullDataTag*>(ptr)) { }
    virtual uint32_t getDataType() const { return tag->data_type; }
    virtual uint32_t getNumberOfBytes() const { return tag->number_of_bytes; }
    virtual size_t   getSizeOf() const { return tag->number_of_bytes+TAG_SIZE; }
    bool isSmall() const { return (tag->hbytes[2]!=0 || tag->hbytes[3]!=0); }
  };
  
  struct SmallDataElement : public DataElement {
    const SmallDataTag *tag;
  public:
    SmallDataElement(const char *ptr) :
      DataElement(ptr+SMALL_TAG_SIZE),
    tag(reinterpret_cast<const SmallDataTag*>(ptr)) { }
    virtual uint32_t getDataType() const { return static_cast<uint32_t>(tag->data_type); }
    virtual uint32_t getNumberOfBytes() const { return static_cast<uint32_t>(tag->number_of_bytes); }
    virtual size_t   getSizeOf() const { return TAG_SIZE; }
  };

  /********************************************/
  
  /************** PROPERTIES *****************/
  char   *mmapped_data;
  size_t  mmapped_data_size;
  size_t  mmapped_data_pos;
  int     fd;
  const HeaderType *header;
  bool swap_bytes;
  april_utils::list<char*> decompressed_buffers;
  
  /** MMAPPED DATA POINTER ACCESS **/
  template<typename T>
  T getCurrent() { return reinterpret_cast<T>(mmapped_data+mmapped_data_pos); }
  void addToCurrent(size_t p) {
    mmapped_data_pos += p;
    april_assert(mmapped_data_pos <= mmapped_data_size);
  }
  /*********************************/
  
public:
  
  MatFileReader(const char *path);
  ~MatFileReader();
  DataElement *getNextDataElement();
  void reset() { mmapped_data_pos = 0; }

  static DataElement *getDataElement(const char *ptr);
  
};

#endif // MATLAB_H
