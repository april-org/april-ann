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
#include <cstdio>
#include <zlib.h>
#include "matlab.h"
#include "endianism.h"
#include "ignore_result.h"
#include "error_print.h"

#define MOD8(v) (v)&(0x07)

using april_utils::swap_bytes_in_place;
using april_utils::string;

int inflate(const void *src, int srcLen, void *dst, int dstLen) {
  z_stream strm  = {0};
  strm.total_in  = strm.avail_in  = srcLen;
  strm.total_out = strm.avail_out = dstLen;
  strm.next_in   = (Bytef *) src;
  strm.next_out  = (Bytef *) dst;
  
  strm.zalloc = Z_NULL;
  strm.zfree  = Z_NULL;
  strm.opaque = Z_NULL;

  int err = -1;
  int ret = -1;

  // 15 window bits, and the +32 tells zlib to to detect if using gzip or zlib
  err = inflateInit2(&strm, (15 + 32));
  if (err == Z_OK) {
    err = inflate(&strm, Z_FINISH);
    if (err == Z_STREAM_END) {
      ret = strm.total_out;
    }
    else {
      inflateEnd(&strm);
      return err;
    }
  }
  else {
    inflateEnd(&strm);
    return err;
  }
  inflateEnd(&strm);
  return ret;
}

template<typename T>
void readMatrixData(MatrixFloat::col_major_iterator &m_it,
		    MatrixFloat::col_major_iterator &end,
		    const T *ptr, const uint32_t nbytes) {
  for (uint32_t ptr_pos=0; ptr_pos < nbytes; ptr_pos += sizeof(T), ++ptr) {
    april_assert(m_it != end);
    *m_it = static_cast<float>(*ptr);
    ++m_it;
  }
}

template<typename T>
void readMatrixData(MatrixChar::col_major_iterator &m_it,
		    MatrixChar::col_major_iterator &end,
		    const T *ptr, const uint32_t nbytes) {
  for (uint32_t ptr_pos=0; ptr_pos < nbytes; ptr_pos += sizeof(T), ++ptr) {
    april_assert(m_it != end);
    *m_it = *ptr;
    ++m_it;
  }
}

template<typename T>
void readMatrixData(MatrixInt32::col_major_iterator &m_it,
		    MatrixInt32::col_major_iterator &end,
		    const T *ptr, const uint32_t nbytes) {
  for (uint32_t ptr_pos=0; ptr_pos < nbytes; ptr_pos += sizeof(T), ++ptr) {
    april_assert(m_it != end);
    *m_it = *ptr;
    ++m_it;
  }
}

///////////////////////// TaggedDataElement //////////////////////////////////

MatFileReader::TaggedDataElement *MatFileReader::TaggedDataElement::
getNextSubElement() {
  if (data_pos < getSizeOf() - TAG_SIZE) {
    TaggedDataElement *sub_element = getDataElement(data + data_pos);
    data_pos += sub_element->getSizeOf();
    // FOR THE PADDING OF 64 BITS (8 BYTES)
    size_t mod8 = MOD8(data_pos);
    if (mod8 != 0)  data_pos += (8 - mod8);
    april_assert(data_pos <= getSizeOf());
    return sub_element;
  }
  else return 0;
}

uint32_t MatFileReader::TaggedDataElement::getClass() {
  if (getDataType() != MATRIX) {
    ERROR_PRINT1("Impossible to get CLASS from a non "
		 "Matrix element (type %d)\n", getDataType());
    return 0;
  }
  int old_data_pos = data_pos;
  TaggedDataElement *array_flags = getNextSubElement();
  data_pos = old_data_pos;
  const char *flags = array_flags->getData<const char*>();
  // CHECK: [0] only for little_endian
  uint32_t cl = static_cast<uint32_t>(flags[0]);
  delete array_flags;
  return cl;
}

MatFileReader::CellArrayDataElement *MatFileReader::TaggedDataElement::
getCellArray(char *name, size_t maxsize) {
  if (getDataType() != MATRIX) {
    ERROR_PRINT1("Impossible to get a Matrix from a non "
		 "Matrix element (type %d)\n", getDataType());
    return 0;
  }
  if (getClass() != CL_CELL_ARRAY) {
    ERROR_PRINT1("Impossible to get a cell array from a non cell array "
		 "element (class %d)\n", getClass());
    return 0;
  }
  TaggedDataElement *array_flags = getNextSubElement();
  TaggedDataElement *dims_array  = getNextSubElement();
  TaggedDataElement *array_name  = getNextSubElement();
  if (array_flags==0 || dims_array==0 || array_name==0)
    ERROR_EXIT(128, "Impossible to found all ARRAY elements\n");
  int num_dims = dims_array->getNumberOfBytes()/sizeof(int32_t);
  const int32_t *dims = dims_array->getData<const int32_t*>();
  CellArrayDataElement *cell_array = new CellArrayDataElement(dims, num_dims);
  TaggedDataElement *cell;
  int idx=0;
  while( (cell = getNextSubElement()) != 0 && idx < cell_array->getSize())
    cell_array->setElementAt(idx++, cell);
  if (array_name->getNumberOfBytes() > maxsize-1)
    ERROR_EXIT2(128, "Overflow at array name, found %u, maximum %lu\n",
		array_name->getNumberOfBytes(), maxsize);
  strncpy(name, array_name->getData<const char*>(),
	  array_name->getNumberOfBytes());
  name[array_name->getNumberOfBytes()] = '\0';
  delete array_flags;
  delete dims_array;
  delete array_name;
  return cell_array;
}

MatFileReader::StructureDataElement *MatFileReader::TaggedDataElement::
getStructure(char *name, size_t maxsize) {
  if (getDataType() != MATRIX) {
    ERROR_PRINT1("Impossible to get a Matrix from a non "
		 "Matrix element (type %d)\n", getDataType());
    return 0;
  }
  if (getClass() != CL_STRUCTURE) {
    ERROR_PRINT1("Impossible to get a structure from a non structure "
		 "element (class %d)\n", getClass());
    return 0;
  }
  TaggedDataElement *array_flags  = getNextSubElement();
  TaggedDataElement *dims_array   = getNextSubElement();
  TaggedDataElement *array_name   = getNextSubElement();
  TaggedDataElement *fname_length = getNextSubElement();
  TaggedDataElement *fnames       = getNextSubElement();
  if (array_flags==0 || dims_array==0 || array_name==0 ||
      fname_length == 0 || fnames == 0)
    ERROR_EXIT(128, "Impossible to found all ARRAY elements\n");
  StructureDataElement *structure = new StructureDataElement();
  TaggedDataElement *field;
  // size of every name
  int32_t fname_length_value = fname_length->getData<const int32_t*>()[0];
  // compute the number of fields (size)
  int size = fnames->getNumberOfBytes()/fname_length_value;
  // pointer to the current name
  const char *current_name=fnames->getData<const char*>();
  for (int i=0; i<size; ++i) {
    field = getNextSubElement();
    if (field == 0)
      ERROR_EXIT2(128, "Incorrect number of fields, found %d, expected %d\n",
		  i, size);
    structure->setElementByName(current_name, field);
    current_name += fname_length_value;
  }
  if (array_name->getNumberOfBytes() > maxsize-1)
    ERROR_EXIT2(128, "Overflow at array name, found %u, maximum %lu\n",
		array_name->getNumberOfBytes(), maxsize);
  strncpy(name, array_name->getData<const char*>(),
	  array_name->getNumberOfBytes());
  name[array_name->getNumberOfBytes()] = '\0';
  delete array_flags;
  delete dims_array;
  delete array_name;
  delete fname_length;
  delete fnames;
  return structure;
}

// ONLY IF CHAR ARRAY
MatrixChar *MatFileReader::TaggedDataElement::
getMatrixChar(char *name, size_t maxsize) {
  if (getDataType() != MATRIX) {
    ERROR_PRINT1("Impossible to get a Matrix from a non "
		 "Matrix element (type %d)\n", getDataType());
    return 0;
  }
  uint32_t cl = getClass();
  if (cl != CL_CHAR) {
    ERROR_PRINT1("Incorrect array class type: %d\n", cl);
    return 0;
  }
  TaggedDataElement *array_flags = getNextSubElement();
  TaggedDataElement *dims_array  = getNextSubElement();
  TaggedDataElement *array_name  = getNextSubElement();
  TaggedDataElement *real_part   = getNextSubElement();
  if (array_flags==0 || dims_array==0 || array_name==0 || real_part==0)
    ERROR_EXIT(128, "Impossible to found all ARRAY elements\n");
  int num_dims = dims_array->getNumberOfBytes()/sizeof(int32_t);
  int *dims = new int[num_dims+1];
  const int32_t *const_dims = dims_array->getData<const int32_t*>();
  for (int i=0; i<num_dims; ++i)
    dims[i] = static_cast<int>(const_dims[i]);
  MatrixChar *m;
  m = new MatrixChar(num_dims, dims);
  delete[] dims;
  // traversing in col_major the real/img part of the matrix will be traversed
  // last, so it is possible add all real components in a first step, and in a
  // second step to add all the imaginary components
  MatrixChar::col_major_iterator it(m->begin());
  MatrixChar::col_major_iterator end(m->end());
  // this loop traverses all the real components, and later all the imaginary
  // components (if any)
  switch(real_part->getDataType()) {
  case UTF8:
    readMatrixData(it, end, real_part->getData<const char*>(),
		   real_part->getNumberOfBytes());
    break;
  case UTF16:
  case UTF32:
    ERROR_EXIT(128, "UTF16 and UTF32 formats are not implemented\n");
    break;
  default:
    ERROR_EXIT1(128, "Data type %d not supported, use only UTF8\n",
		real_part->getDataType());
  }
  if (array_name->getNumberOfBytes() > maxsize-1)
    ERROR_EXIT2(128, "Overflow at array name, found %u, maximum %lu\n",
		array_name->getNumberOfBytes(), maxsize);
  strncpy(name, array_name->getData<const char*>(),
	  array_name->getNumberOfBytes());
  name[array_name->getNumberOfBytes()] = '\0';
  delete array_flags;
  delete dims_array;
  delete array_name;
  delete real_part;
  return m;  
}

MatrixFloat *MatFileReader::TaggedDataElement::getMatrix(char *name,
							 size_t maxsize,
							 bool col_major) {
  if (getDataType() != MATRIX) {
    ERROR_PRINT1("Impossible to get a Matrix from a non "
		 "Matrix element (type %d)\n", getDataType());
    return 0;
  }
  uint32_t cl = getClass();
  switch(cl) {
  case CL_CELL_ARRAY:
  case CL_STRUCTURE:
  case CL_OBJECT:
  case CL_CHAR:
  case CL_SPARSE:
    ERROR_PRINT1("Incorrect array class type: %d\n", cl);
    return 0;
  default:
    ;
  }
  TaggedDataElement *array_flags = getNextSubElement();
  TaggedDataElement *dims_array  = getNextSubElement();
  TaggedDataElement *array_name  = getNextSubElement();
  TaggedDataElement *real_part   = getNextSubElement();
  TaggedDataElement *img_part    = getNextSubElement();
  if (array_flags==0 || dims_array==0 || array_name==0 || real_part==0)
    ERROR_EXIT(128, "Impossible to found all ARRAY elements\n");
  int num_dims = dims_array->getNumberOfBytes()/sizeof(int32_t);
  int *dims = new int[num_dims+1];
  const int32_t *const_dims = dims_array->getData<const int32_t*>();
  for (int i=0; i<num_dims; ++i)
    dims[i] = static_cast<int>(const_dims[i]);
  // if it has imaginary part, the resulting matrix will had another last
  // dimension of size 2, where position 0 contains the real part, and position
  // 1 the imaginary part
  if (img_part != 0) {
    dims[num_dims] = 2;
  }
  else dims[num_dims] = 1;
  MatrixFloat *m;
  m = new MatrixFloat(num_dims+((img_part!=0)?1:0), dims,
		      (col_major)?CblasColMajor:CblasRowMajor);
  // traversing in col_major the real/img part of the matrix will be traversed
  // last, so it is possible add all real components in a first step, and in a
  // second step to add all the imaginary components
  MatrixFloat::col_major_iterator it(m->begin());
  MatrixFloat::col_major_iterator end(m->end());
  TaggedDataElement *numeric_elems[2] = { real_part, img_part };
  // this loop traverses all the real components, and later all the imaginary
  // components (if any)
  for (int i=0; i<dims[num_dims]; ++i) {
    switch(numeric_elems[i]->getDataType()) {
    case SINGLE:
      readMatrixData(it, end, numeric_elems[i]->getData<const float*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case DOUBLE:
      readMatrixData(it, end, numeric_elems[i]->getData<const double*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case INT8:
      readMatrixData(it, end, numeric_elems[i]->getData<const int8_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case UINT8:
      readMatrixData(it, end, numeric_elems[i]->getData<const uint8_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case INT16:
      readMatrixData(it, end, numeric_elems[i]->getData<const int16_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case UINT16:
      readMatrixData(it, end, numeric_elems[i]->getData<const uint16_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case INT32:
    case UINT32:
    case INT64:
    case UINT64:
    default:
      ERROR_EXIT1(128,"Data type %d not supported\n",
		  numeric_elems[i]->getDataType());
    }
  }
  if (array_name->getNumberOfBytes() > maxsize-1)
    ERROR_EXIT2(128, "Overflow at array name, found %u, maximum %lu\n",
		array_name->getNumberOfBytes(), maxsize);
  strncpy(name, array_name->getData<const char*>(),
	  array_name->getNumberOfBytes());
  name[array_name->getNumberOfBytes()] = '\0';
  delete array_flags;
  delete dims_array;
  delete array_name;
  delete real_part;
  delete img_part;
  delete[] dims;
  return m;
}

MatrixInt32 *MatFileReader::TaggedDataElement::getMatrixInt32(char *name,
							      size_t maxsize) {
  if (getDataType() != MATRIX) {
    ERROR_PRINT1("Impossible to get a Matrix from a non "
		 "Matrix element (type %d)\n", getDataType());
    return 0;
  }
  uint32_t cl = getClass();
  switch(cl) {
  case CL_CELL_ARRAY:
  case CL_STRUCTURE:
  case CL_OBJECT:
  case CL_CHAR:
  case CL_SPARSE:
    ERROR_PRINT1("Incorrect array class type: %d\n", cl);
    return 0;
  default:
    ;
  }
  TaggedDataElement *array_flags = getNextSubElement();
  TaggedDataElement *dims_array  = getNextSubElement();
  TaggedDataElement *array_name  = getNextSubElement();
  TaggedDataElement *real_part   = getNextSubElement();
  TaggedDataElement *img_part    = getNextSubElement();
  if (array_flags==0 || dims_array==0 || array_name==0 || real_part==0)
    ERROR_EXIT(128, "Impossible to found all ARRAY elements\n");
  int num_dims = dims_array->getNumberOfBytes()/sizeof(int32_t);
  int *dims = new int[num_dims+1];
  const int32_t *const_dims = dims_array->getData<const int32_t*>();
  for (int i=0; i<num_dims; ++i)
    dims[i] = static_cast<int>(const_dims[i]);
  // if it has imaginary part, the resulting matrix will had another last
  // dimension of size 2, where position 0 contains the real part, and position
  // 1 the imaginary part
  if (img_part != 0) {
    dims[num_dims] = 2;
  }
  else dims[num_dims] = 1;
  MatrixInt32 *m;
  m = new MatrixInt32(num_dims+((img_part!=0)?1:0), dims);
  delete[] dims;
  // traversing in col_major the real/img part of the matrix will be traversed
  // last, so it is possible add all real components in a first step, and in a
  // second step to add all the imaginary components
  MatrixInt32::col_major_iterator it(m->begin());
  MatrixInt32::col_major_iterator end(m->end());
  TaggedDataElement *numeric_elems[2] = { real_part, img_part };
  // this loop traverses all the real components, and later all the imaginary
  // components (if any)
  for (int i=0; i<dims[num_dims]; ++i) {
    switch(numeric_elems[i]->getDataType()) {
    case INT8:
      readMatrixData(it, end, numeric_elems[i]->getData<const int8_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case UINT8:
      readMatrixData(it, end, numeric_elems[i]->getData<const uint8_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case INT16:
      readMatrixData(it, end, numeric_elems[i]->getData<const int16_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case UINT16:
      readMatrixData(it, end, numeric_elems[i]->getData<const uint16_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case INT32:
      readMatrixData(it, end, numeric_elems[i]->getData<const int32_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case UINT32:
      readMatrixData(it, end, numeric_elems[i]->getData<const uint32_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case INT64:
      readMatrixData(it, end, numeric_elems[i]->getData<const int64_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case UINT64:
      readMatrixData(it, end, numeric_elems[i]->getData<const uint64_t*>(),
		     numeric_elems[i]->getNumberOfBytes());
      break;
    case SINGLE:
    case DOUBLE:
    default:
      ERROR_EXIT1(128,"Data type %d not supported\n",
		  numeric_elems[i]->getDataType());
    }
  }
  if (array_name->getNumberOfBytes() > maxsize-1)
    ERROR_EXIT2(128, "Overflow at array name, found %u, maximum %lu\n",
		array_name->getNumberOfBytes(), maxsize);
  strncpy(name, array_name->getData<const char*>(),
	  array_name->getNumberOfBytes());
  name[array_name->getNumberOfBytes()] = '\0';
  delete array_flags;
  delete dims_array;
  delete array_name;
  delete real_part;
  delete img_part;
  return m;
}

//////////////////////// CellArrayDataElement //////////////////////////////////
MatFileReader::CellArrayDataElement::
CellArrayDataElement(const int *dims, int num_dims) :
  DataElementInterface(),
  dims(new int[num_dims]), stride(new int[num_dims]), num_dims(num_dims),
  total_size(1) {
  for (int i=0; i<num_dims; ++i) {
    this->dims[i]   = dims[i];
    this->stride[i] = total_size;
    total_size     *= dims[i];
  }
  elements = new TaggedDataElement*[total_size];
  for (int i=0; i<total_size; ++i) elements[i] = 0;
}

MatFileReader::CellArrayDataElement::
~CellArrayDataElement() {
  delete[] dims;
  delete[] stride;
  for (int i=0; i<total_size; ++i) if (elements[i] != 0) DecRef(elements[i]);
  delete[] elements;
}

int MatFileReader::CellArrayDataElement::computeRawPos(const int *coords) {
  int raw_pos = 0;
  for (int i=0; i<num_dims; ++i) raw_pos += stride[i]*coords[i];
  return raw_pos;
}

void MatFileReader::CellArrayDataElement::setElementAt(const int *coords, int n,
						       TaggedDataElement *e) {
  if (n != num_dims)
    ERROR_EXIT2(128, "Incorrect size of coords vector, found %d, expected %d\n",
		n, num_dims);
  setElementAt(computeRawPos(coords), e);
}

void MatFileReader::CellArrayDataElement::
setElementAt(int raw_idx, TaggedDataElement *e) {
  assert(raw_idx >= 0 && raw_idx < total_size);
  if (elements[raw_idx] != 0)
    ERROR_EXIT(128, "The given position is not empty\n");
  elements[raw_idx] = e;
  IncRef(e);
}

MatFileReader::TaggedDataElement *MatFileReader::CellArrayDataElement::
getElementAt(const int *coords, int n) {
  if (n != num_dims)
    ERROR_EXIT2(128, "Incorrect size of coords vector, found %d, expected %d\n",
		n, num_dims);
  return getElementAt(computeRawPos(coords));
}

MatFileReader::TaggedDataElement *MatFileReader::CellArrayDataElement::
getElementAt(int raw_idx) {
  assert(raw_idx >= 0 && raw_idx < total_size);
  return elements[raw_idx];
}

//////////////////////// StructureDataElement //////////////////////////////////
MatFileReader::StructureDataElement::StructureDataElement() :
  DataElementInterface() {
}

MatFileReader::StructureDataElement::
~StructureDataElement() {
  for (HashType::iterator it=elements.begin(); it!=elements.end(); ++it)
    DecRef(it->second);
}

void MatFileReader::StructureDataElement::
setElementByName(const char *name, TaggedDataElement *e) {
  string key(name);
  TaggedDataElement *&value = elements[key];
  if (value != 0)
    ERROR_EXIT1(128, "The given name '%s' is occupied\n", name);
  value = e;
  IncRef(value);
}

MatFileReader::TaggedDataElement *MatFileReader::StructureDataElement::
getElementByName(const char *name) {
  string key(name);
  return elements[key];
}

//////////////////////////// MatFileReader /////////////////////////////////////

MatFileReader::MatFileReader(const char *path) :
  Referenced(), mmapped_data_pos(0) {
  if ((fd = open(path, O_RDONLY)) < 0)
    ERROR_EXIT1(128,"Unable to open  file %s\n", path);
  // find size of input file
  struct stat statbuf;
  if (fstat(fd, &statbuf) < 0) {
    ERROR_PRINT("Error guessing filesize\n");
    exit(1);
  }
  // mmap the input file
  mmapped_data_size = statbuf.st_size;
  if ((mmapped_data = static_cast<char*>(mmap(0, mmapped_data_size,
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
    ERROR_EXIT(128, "ENDIANISM LOGIC NOT IMPLEMENTED\n");
}

MatFileReader::~MatFileReader() {
  munmap(mmapped_data, mmapped_data_size);
  close(fd);
  while(!decompressed_buffers.empty()) {
    delete[] decompressed_buffers.front();
    decompressed_buffers.pop_front();
  }
}

MatFileReader::TaggedDataElement *MatFileReader::getNextDataElement() {
  if (mmapped_data_pos < mmapped_data_size) {
    TaggedDataElement *element = getDataElement(getCurrent<char*>());
    if (element->getDataType()!=MATRIX && element->getDataType()!=COMPRESSED)
      ERROR_EXIT1(128, "Incorrect data type, expected MATRIX or "
		  "COMPRESSED, found %d\n", element->getDataType());
    addToCurrent(element->getSizeOf());
    if (element->getDataType() == COMPRESSED) {
      int len    = static_cast<int>(element->getNumberOfBytes());
      char *buff = 0;
      int err;
      do {
	len = len << 1;
	delete[] buff;
	buff = new char[len];
      } while((err=inflate(element->getData<const void*>(),
			   element->getNumberOfBytes(),
			   reinterpret_cast<void*>(buff),len)) == Z_BUF_ERROR);
      delete element;
      element = getDataElement(buff);
      decompressed_buffers.push_front(buff);
    }
    return element; 
  }
  else return 0;
}

MatFileReader::TaggedDataElement *MatFileReader::
getDataElement(const char *ptr)
{
  TaggedDataElement *ret;
  FullTagDataElement *full_data;
  full_data = new FullTagDataElement(ptr);
  if (full_data->isSmall()) {
    delete full_data;
    ret = new SmallTagDataElement(ptr);
  }
  else ret = full_data;
  return ret;
}
