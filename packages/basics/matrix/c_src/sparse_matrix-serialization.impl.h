/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
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
#ifndef SPARSE_MATRIX_SERIALIZATION_H
#define SPARSE_MATRIX_SERIALIZATION_H
#include <cmath>
#include <cstdio>
#include <cstring>

#include "binarizer.h"
#include "constString.h"
#include "c_string.h"
#include "error_print.h"
#include "matrix.h"
#include "matrix_serialization_utils.h"
#include "smart_ptr.h"
#include "stream.h"

namespace Basics {

  /////////////////////////////////////////////////////////////////////////////

  // SPARSE

  template <typename T>
  SparseMatrix<T>*
  SparseMatrix<T>::read(AprilIO::StreamInterface *stream,
                        const AprilUtils::LuaTable &options) {
    UNUSED_VARIABLE(options);
    MatrixIO::AsciiExtractor<T> ascii_extractor;
    MatrixIO::BinaryExtractor<T> bin_extractor;
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    AprilUtils::SharedPtr<AprilIO::CStringStream>
      c_str(new AprilIO::CStringStream());
    april_assert(!c_str.empty());
    AprilUtils::constString line,format,sparse,token;
    // First we read the matrix dimensions
    line = readULine(stream, c_str.get());
    if (!line) {
      ERROR_PRINT("empty file!!!\n");
      return 0;
    }
    int dims[2], n=0, NZ;
    for (int i=0; i<2; ++i, ++n) {
      if (!line.extract_int(&dims[i])) {
        ERROR_PRINT1("incorrect dimension %d type, expected a integer\n", n);
        return 0;
      }
      n++;
    }
    if (!line.extract_int(&NZ)) {
      ERROR_PRINT("impossible to read the number of non-zero elements\n");
      return 0;
    }
    SparseMatrix<T> *mat = 0;
    // Now we read the type of the format
    line = readULine(stream, c_str.get());
    format = line.extract_token();
    if (!format) {
      ERROR_PRINT("impossible to read format token\n");
      return 0;
    }
    sparse = line.extract_token();
    AprilMath::GPUMirroredMemoryBlock<T> *values = new AprilMath::GPUMirroredMemoryBlock<T>(NZ);
    AprilMath::Int32GPUMirroredMemoryBlock *indices = new AprilMath::Int32GPUMirroredMemoryBlock(NZ);
    AprilMath::Int32GPUMirroredMemoryBlock *first_index = 0;
    if (!sparse || sparse=="csr") {
      first_index = new AprilMath::Int32GPUMirroredMemoryBlock(dims[0]+1);
    }
    else if (sparse=="csc") {
      first_index = new AprilMath::Int32GPUMirroredMemoryBlock(dims[1]+1);
    }
    else {
      ERROR_PRINT("Impossible to determine the sparse format\n");
      return 0;
    }
    T *values_ptr = values->getPPALForWrite();
    int32_t *indices_ptr = indices->getPPALForWrite();
    int32_t *first_index_ptr = first_index->getPPALForWrite();
    if (format == "ascii") {
      int i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              ascii_extractor(line, values_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              line.extract_int(&indices_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<static_cast<int>(first_index->getSize())) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<static_cast<int>(first_index->getSize()) &&
              line.extract_int(&first_index_ptr[i])) {
          ++i;
        }
      }
    } else { // binary
      int i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              bin_extractor(line, values_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<NZ) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<NZ &&
              line.extract_int32_binary(&indices_ptr[i])) {
          ++i;
        }
      }
      i=0;
      while(i<static_cast<int>(first_index->getSize())) {
        if (! (line=readULine(stream, c_str.get())) ) {
          ERROR_EXIT(128, "Incorrect sparse matrix format\n");
        }
        while(i<static_cast<int>(first_index->getSize()) &&
              line.extract_int32_binary(&first_index_ptr[i])) {
          ++i;
        }
      }
    }
    if (sparse=="csr") {
      mat = new SparseMatrix<T>(dims[0],dims[1],
                                values,indices,first_index,
                                CSR_FORMAT);
    }
    else {
      // This was checked before: else if (sparse=="csc") {
      mat = new SparseMatrix<T>(dims[0],dims[1],
                                values,indices,first_index,
                                CSC_FORMAT);
    }
    return mat;
  }
  
  template <typename T>
  void SparseMatrix<T>::write(AprilIO::StreamInterface *stream,
                              const AprilUtils::LuaTable &options) {
    bool is_ascii = options.opt(MatrixIO::ASCII_OPTION, false);
    //
    MatrixIO::SparseAsciiSizer<T> ascii_sizer;
    MatrixIO::SparseBinarySizer<T> bin_sizer;
    MatrixIO::AsciiCoder<T> ascii_coder;
    MatrixIO::BinaryCoder<T> bin_coder;
    int sizedata,sizeheader;
    sizeheader = (this->getNumDim()+1)*10+10+10; // FIXME: To put adequate values
    // sizedata contains the memory used by T in ascii including spaces,
    // new lines, etc...
    if (is_ascii) {
      sizedata = ascii_sizer(this) +
        this->nonZeroSize()*12 +
        this->getDenseCoordinateSize()*12 + 3;
    }
    else {
      sizedata = bin_sizer(this) +
        AprilUtils::binarizer::buffer_size_32(this->nonZeroSize()) +
        AprilUtils::binarizer::buffer_size_32(this->getDenseCoordinateSize()) +
        3;
    }
    // size_t expected_size = static_cast<size_t>(sizedata+sizeheader+1);
    // UNUSED_VARIABLE(expected_size);
    if (!stream->isOpened()) {
      ERROR_EXIT(256, "The stream is not prepared\n");
    }
    stream->printf("%d ",this->getDimSize(0));
    stream->printf("%d ",this->getDimSize(1));
    stream->printf("%d\n",this->nonZeroSize());
    const T *values_ptr = this->getRawValuesAccess()->getPPALForRead();
    const int32_t *indices_ptr = this->getRawIndicesAccess()->getPPALForRead();
    const int32_t *first_index_ptr = this->getRawFirstIndexAccess()->getPPALForRead();
    if (is_ascii) {
      const int columns = 9;
      stream->printf("ascii");
      if (this->getSparseFormat() == CSR_FORMAT) stream->printf(" csr");
      else stream->printf(" csc");
      stream->printf("\n");
      int i;
      for (i=0; i<this->nonZeroSize(); ++i) {
        ascii_coder(values_ptr[i], stream);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) {
        stream->printf("\n"); 
      }
      for (i=0; i<this->nonZeroSize(); ++i) {
        stream->printf("%d", indices_ptr[i]);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) {
        stream->printf("\n"); 
      }
      for (i=0; i<=this->getDenseCoordinateSize(); ++i) {
        stream->printf("%d", first_index_ptr[i]);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) stream->printf("\n"); 
    } else { // binary
      const int columns = 16;
      stream->printf("binary");
      if (this->getSparseFormat() == CSR_FORMAT)
        stream->printf(" csr");
      else
        stream->printf(" csc");
      stream->printf("\n");
      int i=0;
      char b[5];
      for (i=0; i<this->nonZeroSize(); ++i) {
        bin_coder(values_ptr[i], stream);
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
      for (i=0; i<this->nonZeroSize(); ++i) {
        AprilUtils::binarizer::code_int32(indices_ptr[i], b);
        stream->printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
      for (i=0; i<=this->getDenseCoordinateSize(); ++i) {
        AprilUtils::binarizer::code_int32(first_index_ptr[i], b);
        stream->printf("%c%c%c%c%c", b[0],b[1],b[2],b[3],b[4]);
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
    }
  }

} // namespace Basics

#endif // SPARSE_MATRIX_SERIALIZATION_H
