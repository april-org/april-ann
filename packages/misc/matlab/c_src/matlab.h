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

/******************************************************************
 *
 * Readers of MAT format of Matlab
 * http://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf
 *
 ******************************************************************/

#ifndef MATLAB_H
#define MATLAB_H

#include <cstdio>
#include <stdint.h>
#include "matrixFloat.h"
#include "matrixComplexF.h"
#include "matrixChar.h"
#include "matrixInt32.h"
#include "matrixDouble.h"
#include "list.h"
#include "hash_table.h"
#include "mystring.h"

#define MAX_NAME_SIZE    256
#define HEADER_SIZE      128
#define HEADER_TEXT_SIZE 124
#define TAG_SIZE         8
#define SMALL_TAG_SIZE   4
#define SMALL_DATA_SIZE  4
#define MAX_NUM_DIMS     32

#define DATA_TYPE 0
#define NUMBER_OF_BYTES 1

namespace Matlab {

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
      CL_DOUBLE=6, CL_SINGLE=7, CL_INT8=8, CL_UINT8=9,
      CL_INT16=10, CL_UINT16=11, CL_INT32=12, CL_UINT32=13,
      CL_INT64=14, CL_UINT64=15,
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
  
    class DataElementInterface : public Referenced {
    protected:
      MatFileReader *reader;
    public:
      DataElementInterface(MatFileReader *reader) : Referenced(),reader(reader) {
        IncRef(reader);
      }
      virtual ~DataElementInterface() {
        DecRef(reader);
      }
      virtual uint32_t getDataType() const = 0;
      virtual uint32_t getClass() = 0;
    };

    // forward declaration
    class CellArrayDataElement;
    class StructureDataElement;
  
    class TaggedDataElement : public DataElementInterface {
      friend class MatFileReader;
    protected:
      const char *data;
      size_t data_pos;
      TaggedDataElement(MatFileReader *reader, const char *ptr) :
        DataElementInterface(reader), data(ptr), data_pos(0) { }
    public:
      virtual ~TaggedDataElement() { }
      virtual uint32_t getDataType() const = 0;
      virtual uint32_t getNumberOfBytes() const = 0;
      virtual size_t   getSizeOf() const = 0;
      template<typename T>
      T getData() const { return reinterpret_cast<T>(data); }
      void reset() { data_pos = 0; }
      // SUB-ELEMENTS
      TaggedDataElement *getNextSubElement();
      // FOR NUMERIC TYPES (casting)
      Basics::MatrixFloat *getMatrix(char *name, size_t maxsize, bool col_major=false);
      // FOR NUMERIC TYPES (casting)
      Basics::MatrixComplexF *getMatrixComplexF(char *name, size_t maxsize,
                                                bool col_major=false);
      // FOR NUMERIC TYPES (casting)
      Basics::MatrixDouble *getMatrixDouble(char *name, size_t maxsize);
      virtual uint32_t getClass();
      // ONLY IF CELL ARRAY
      CellArrayDataElement *getCellArray(char *name, size_t maxsize);
      // ONLY IF STRUCTURE
      StructureDataElement *getStructure(char *name, size_t maxsize);
      // ONLY IF CHAR ARRAY
      Basics::MatrixChar *getMatrixChar(char *name, size_t maxsize);
      // FOR NUMERIC TYPES (casting)
      Basics::MatrixInt32 *getMatrixInt32(char *name, size_t maxsize);
    };

    class CellArrayDataElement : public DataElementInterface {
      int *dims, *stride, num_dims, total_size;
      TaggedDataElement **elements;
      int computeRawPos(const int *coords);
    public:
      CellArrayDataElement(MatFileReader *reader, const int *dims, int num_dims);
      ~CellArrayDataElement();
      TaggedDataElement *getElementAt(const int *coords, int n);
      TaggedDataElement *getElementAt(int raw_idx);
      void setElementAt(const int *coords, int n, TaggedDataElement *e);
      void setElementAt(int raw_idx, TaggedDataElement *e);
      int getNumDim() const { return num_dims; }
      int getDimSize(int i) const { return dims[i]; }
      const int *getDimPtr() const { return dims; }
      int getSize() const { return total_size; }
      virtual uint32_t getDataType() const { return MATRIX; }
      virtual uint32_t getClass() { return CL_CELL_ARRAY; }
      void computeCoords(int raw_pos, int *coords) {
        for (int i=num_dims-1; i>=0; --i) {
          coords[i] = raw_pos / stride[i];
          raw_pos = raw_pos % stride[i];
        }
      }
    };

    class StructureDataElement : public DataElementInterface {
    public:
      typedef AprilUtils::hash<AprilUtils::string, TaggedDataElement*> HashType;
    private:
      HashType elements;
    public:
      StructureDataElement(MatFileReader *reader);
      ~StructureDataElement();
      TaggedDataElement *getElementByName(const char *name);
      void setElementByName(const char *name, TaggedDataElement *e);
      virtual uint32_t getDataType() const { return MATRIX; }
      virtual uint32_t getClass() { return CL_STRUCTURE; }
      HashType::iterator begin() { return elements.begin(); }
      HashType::iterator end() { return elements.end(); }
      unsigned int size() const { return elements.size(); }
    };
  
  private:
    struct FullTagDataElement : public TaggedDataElement {
      const FullDataTag *tag;
    public:
      FullTagDataElement(MatFileReader *reader, const char *ptr) :
        TaggedDataElement(reader, ptr+TAG_SIZE),
        tag(reinterpret_cast<const FullDataTag*>(ptr)) { }
      virtual uint32_t getDataType() const { return tag->data_type; }
      virtual uint32_t getNumberOfBytes() const { return tag->number_of_bytes; }
      virtual size_t   getSizeOf() const { return tag->number_of_bytes+TAG_SIZE; }
      bool isSmall() const { return (tag->hbytes[2]!=0 || tag->hbytes[3]!=0); }
    };
  
    struct SmallTagDataElement : public TaggedDataElement {
      const SmallDataTag *tag;
    public:
      SmallTagDataElement(MatFileReader *reader, const char *ptr) :
        TaggedDataElement(reader, ptr+SMALL_TAG_SIZE),
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
    AprilUtils::list<char*> decompressed_buffers;
  
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
    TaggedDataElement *getNextDataElement();
    void reset() { mmapped_data_pos = 0; }

    static TaggedDataElement *getDataElement(MatFileReader *reader,
                                             const char *ptr);
  
  };

} // namespace Matlab
#endif // MATLAB_H
