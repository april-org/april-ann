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
#ifndef MATRIX_SERIALIZATION_H
#define MATRIX_SERIALIZATION_H
#include <cmath>
#include <cstdio>
#include <cstring>

#include "constString.h"
#include "c_string.h"
#include "error_print.h"
#include "matrix.h"
#include "matrix_serialization_utils.h"
#include "smart_ptr.h"
#include "stream.h"

namespace basics {
  
  /****************************************************************************/

  template <typename T>
  Matrix<T>*
  Matrix<T>::read(AprilIO::StreamInterface *stream,
                  const char *given_order) {
    MatrixIO::AsciiExtractor<T> ascii_extractor;
    MatrixIO::BinaryExtractor<T> bin_extractor;
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::SharedPtr<AprilIO::CStringStream>
      c_str(new AprilIO::CStringStream());;
    april_assert(!c_str.empty());
    april_utils::constString line,format,order,token;
    // First we read the matrix dimensions
    line = readULine(stream, c_str.get());
    if (!line) {
      ERROR_PRINT("empty file!!!\n");
      return 0;
    }
    static const int maxdim=100;
    int dims[maxdim];
    int n=0, pos_comodin=-1;
    while (n<maxdim && (token = line.extract_token())) {
      if (token == "*") {
        if (pos_comodin != -1) {
          // Error, more than one comodin
          ERROR_PRINT("more than one '*' reading a matrix\n");
          return 0;
        }
        pos_comodin = n;
      } else if (!token.extract_int(&dims[n])) {
        ERROR_PRINT1("incorrect dimension %d type, expected a integer\n", n);
        return 0;
      }
      n++;
    }
    if (n==maxdim) {
      ERROR_PRINT("number of dimensions overflow\n");
      return 0; // Maximum allocation problem
    }
    Matrix<T> *mat = 0;
    // Now we read the type of the format
    line = readULine(stream, c_str.get());
    format = line.extract_token();
    if (!format) {
      ERROR_PRINT("impossible to read format token\n");
      return 0;
    }
    order = line.extract_token();
    if (given_order != 0) order = given_order;
    if (pos_comodin == -1) { // Normal version
      if (!order || order=="row_major")
        mat = new Matrix<T>(n,dims);
      else if (order == "col_major")
        mat = new Matrix<T>(n,dims,CblasColMajor);
      else {
        ERROR_PRINT("Impossible to determine the order\n");
        return 0;
      }
      typename Matrix<T>::iterator data_it(mat->begin());
      if (format == "ascii") {
        while (data_it!=mat->end() && (line=readULine(stream, c_str.get()))) {
          while (data_it!=mat->end() && ascii_extractor(line, *data_it)) {
            ++data_it;
          }
        }
      } else { // binary
        while (data_it!=mat->end() && (line=readULine(stream, c_str.get()))) {
          while (data_it!=mat->end() && bin_extractor(line, *data_it)) {
            ++data_it;
          }
        }
      }
      if (data_it!=mat->end()) {
        ERROR_PRINT("Impossible to fill all the matrix components\n");
        delete mat; mat = 0;
      }
    } else { // version with comodin
      int size=0,maxsize=4096;
      T *data = new T[maxsize];
      if (format == "ascii") {
        while ( (line=readULine(stream, c_str.get())) ) {
          while (ascii_extractor(line, data[size])) { 
            size++; 
            if (size == maxsize) { // resize data vector
              T *aux = new T[2*maxsize];
              for (int a=0;a<maxsize;a++) {
                aux[a] = data[a];
              }
              maxsize *= 2;
              delete[] data; data = aux;
            }
          }
        }
      } else { // binary
        while ( (line=readULine(stream, c_str.get())) ) {
          while (bin_extractor(line, data[size])) {
            size++;
            if (size == maxsize) { // resize data vector
              T *aux = new T[2*maxsize];
              for (int a=0;a<maxsize;a++) {
                aux[a] = data[a];
              }
              maxsize *= 2;
              delete[] data; data = aux;
            }
          }
        }
      }
      int sizesincomodin = 1;
      for (int i=0;i<n;i++) {
        if (i != pos_comodin) {
          sizesincomodin *= dims[i];
        }
      }
      if ((size % sizesincomodin) != 0) {
        // Error: The size of the data does not coincide
        ERROR_PRINT("data size is not valid reading a matrix with '*'\n");
        delete[] data; return 0;
      }
      dims[pos_comodin] = size / sizesincomodin;
      if (!order || order == "row_major")
        mat = new Matrix<T>(n,dims);
      else if (order == "col_major")
        mat = new Matrix<T>(n,dims,CblasColMajor);
      int i=0;
      for (typename Matrix<T>::iterator it(mat->begin());
           it!=mat->end();
           ++it,++i)
        *it = data[i];
      delete[] data;
    }
    return mat;
  }
  
  // Returns the number of chars written (there is a '\0' that is not counted)
  template <typename T>
  void Matrix<T>::write(AprilIO::StreamInterface *stream,
                        bool is_ascii) {
    MatrixIO::AsciiSizer<T> ascii_sizer;
    MatrixIO::BinarySizer<T> bin_sizer;
    MatrixIO::AsciiCoder<T> ascii_coder;
    MatrixIO::BinaryCoder<T> bin_coder;
    int sizedata,sizeheader;
    sizeheader = this->getNumDim()*10+10+10; // FIXME: To put adequate values
    // sizedata contains the memory used by T in ascii including spaces,
    // new lines, etc...
    if (is_ascii) sizedata = ascii_sizer(this);
    else sizedata = bin_sizer(this);
    size_t expected_size = static_cast<size_t>(sizedata+sizeheader+1);
    UNUSED_VARIABLE(expected_size);
    if (!stream->isOpened()) {
      ERROR_EXIT(256, "The stream is not prepared\n");
    }
    for (int i=0;i<this->getNumDim()-1;i++) {
      stream->printf("%d ",this->getDimSize(i));
    }
    stream->printf("%d\n",this->getDimSize(this->getNumDim()-1));
    if (is_ascii) {
      const int columns = 9;
      stream->printf("ascii");
      if (this->getMajorOrder() == CblasColMajor) {
        stream->printf(" col_major");
      }
      else {
        stream->printf(" row_major");
      }
      stream->printf("\n");
      int i=0;
      for(typename Matrix<T>::const_iterator it(this->begin());
          it!=this->end();++it,++i) {
        ascii_coder(*it, stream);
        stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
      }
      if ((i % columns) != 0) {
        stream->printf("\n"); 
      }
    } else { // binary
      const int columns = 16;
      stream->printf("binary");
      if (this->getMajorOrder() == CblasColMajor) {
        stream->printf(" col_major");
      }
      else {
        stream->printf(" row_major");
      }
      stream->printf("\n");
      // We substract 1 so the final '\0' is not considered
      int i=0;
      for(typename Matrix<T>::const_iterator it(this->begin());
          it!=this->end();
          ++it,++i) {
        bin_coder(*it, stream);
        /*
          char b[5];
          binarizer::code_float(*it, b);
          fprintf(f, "%c%c%c%c%c", b[0], b[1], b[2], b[3], b[4]);
        */
        if ((i+1) % columns == 0) stream->printf("\n");
      }
      if ((i % columns) != 0) stream->printf("\n"); 
    }
  }

  /****************************************************************/

  template <typename T>
  Matrix<T>*
  Matrix<T>::readTab(AprilIO::StreamInterface *stream, const char *given_order,
                     const char *delim, bool keep_delim, T default_value) {
    if (delim == 0) delim = "\n\r\t ,;";
    MatrixIO::AsciiExtractor<T> ascii_extractor;
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::SharedPtr<AprilIO::CStringStream> c_str;
    c_str = new AprilIO::CStringStream();
    april_assert(!c_str.empty());
    april_utils::constString line("");
    int ncols = 0, nrows = 0;
    while (!stream->eof()) {
      line = readULine(stream, c_str.get(), keep_delim);
      if (line.len() > 0) {
        if (ncols == 0) {
          while(line.extract_token(delim, keep_delim)) ++ncols;
          // while(ascii_extractor(line,value)) ++ncols;
        }
        ++nrows;
      }
    }
    if (nrows <= 0 || ncols <= 0) ERROR_EXIT(256, "Found 0 rows or 0 cols\n");
    stream->seek(SEEK_SET, 0);
    if (stream->hasError()) {
      ERROR_PRINT1("Impossible to rewind the stream: %s\n",
                   stream->getErrorMsg());
      return 0;
    }
    if (!stream->good()) {
      ERROR_PRINT("The stream is not prepared, it is empty, or EOF\n");
      return 0;
    }
    april_utils::constString order( (given_order) ? given_order : "row_major"),token;
    int dims[2] = { nrows, ncols };
    Matrix<T> *mat = 0;
    if (order=="row_major") {
      mat = new Matrix<T>(2,dims);
    }
    else if (order == "col_major") {
      mat = new Matrix<T>(2,dims,CblasColMajor);
    }
    else {
      ERROR_PRINT("Impossible to determine the order\n");
      return 0;
    }
    int i=0;
    typename Matrix<T>::iterator data_it(mat->begin());
    if (keep_delim) {
      // Allows delim at end of the token and therefore empty fields can be
      // identified and assigned to default_value.
      while (data_it!=mat->end() && (line=readULine(stream, c_str.get(), true))) {
        int num_cols_size_count = 0;
        while (data_it!=mat->end()) {
          token = line.extract_token(delim, keep_delim);
          if (!token) break;
          if ( keep_delim && token.len() == 1 &&
               ( strchr(delim, token[0]) || strchr("\r\n", token[0]) ) ) {
            *data_it = default_value;
          }
          else {
            ascii_extractor(token, *data_it);
          }
          ++data_it;
          ++num_cols_size_count;
        }
        if (num_cols_size_count != ncols) {
          ERROR_EXIT3(128, "Incorrect number of elements at line %d, "
                      "expected %d, found %d\n", i, ncols, num_cols_size_count);
        }
        ++i;
      }
    }
    else {
      // Doesn't allow delim at end of the token and empty fields are forbidden
      while (data_it!=mat->end() && (line=readULine(stream, c_str.get()))) {
        int num_cols_size_count = 0;
        while (data_it!=mat->end() &&
               ascii_extractor(line, *data_it)) {
          ++data_it;
          ++num_cols_size_count;
        }
        if (num_cols_size_count != ncols) {
          ERROR_EXIT3(128, "Incorrect number of elements at line %d, "
                      "expected %d, found %d\n", i, ncols, num_cols_size_count);
        }
        ++i;
      }
    }
    if (data_it!=mat->end()) {
      ERROR_PRINT("Impossible to fill all the matrix components\n");
      delete mat; mat = 0;
    }
    return mat;
  }
  
  // Returns the number of chars written (there is a '\0' that is not counted)
  template <typename T>
  void Matrix<T>::writeTab(AprilIO::StreamInterface *stream) {
    MatrixIO::AsciiSizer<T> ascii_sizer;
    MatrixIO::AsciiCoder<T> ascii_coder;
    if (this->getNumDim() != 2) {
      ERROR_EXIT(128, "Needs a matrix with 2 dimensions");
    }
    //
    int sizedata;
    sizedata = ascii_sizer(this);
    size_t expected_size = static_cast<size_t>(sizedata+1);
    UNUSED_VARIABLE(expected_size);
    if (!stream->isOpened()) {
      ERROR_EXIT(256, "The stream is not prepared\n");
    }
    const int columns = this->getDimSize(1);
    int i=0;
    for(typename Matrix<T>::const_iterator it(this->begin());
        it!=this->end();++it,++i) {
      ascii_coder(*it, stream);
      stream->printf("%c", ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      stream->printf("\n"); 
    }
  }
  
} // namespace basics

#endif // MATRIX_SERIALIZATION
