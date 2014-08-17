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
#ifndef MATRIX_SERIALIZATION_UTILS_H
#define MATRIX_SERIALIZATION_UTILS_H
#include "constString.h"
#include "error_print.h"
#include "stream.h"
#include "unused_variable.h"

namespace basics {
  
  template<typename T> class Matrix; // forward declaration
  template<typename T> class SparseMatrix; // forward declaration

  namespace MatrixIO {
    
    /* Templates for ascii and binary extractors, sizers and coders */
  
    /**
     * Template for AsciiExtractor implementations, it needs to be specialized
     * depending in the Matrix<T> type T.
     */
    template<typename T>
    struct AsciiExtractor {
      // returns true if success, false otherwise
      bool operator()(april_utils::constString &line, T &destination) {
        UNUSED_VARIABLE(line);
        UNUSED_VARIABLE(destination);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return false;
      }
    };

    /**
     * Template for BinaryExtractor implementations, it needs to be specialized
     * depending in the Matrix<T> type T.
     */
    template<typename T>
    struct BinaryExtractor {
      // returns true if success, false otherwise
      bool operator()(april_utils::constString &line, T &destination) {
        UNUSED_VARIABLE(line);
        UNUSED_VARIABLE(destination);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return false;
      }
    };

    /**
     * Template for AsciiSizer implementations, it needs to be specialized
     * depending in the Matrix<T> type T.
     */
    template <typename T>
    struct AsciiSizer {
      // returns the number of bytes needed for all matrix data (plus spaces)
      int operator()(const Matrix<T> *mat) {
        UNUSED_VARIABLE(mat);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return -1;
      }
    };

    /**
     * Template for BinarySizer implementations, it needs to be specialized
     * depending in the Matrix<T> type T.
     */
    template <typename T>
    struct BinarySizer {
      // returns the number of bytes needed for all matrix data (plus spaces)
      int operator()(const Matrix<T> *mat) {
        UNUSED_VARIABLE(mat);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return -1;
      }
    };

    /**
     * Template for AsciiCoder implementations, it needs to be specialized
     * depending in the Matrix<T> type T.
     */
    template <typename T>
    struct AsciiCoder {
      // puts to the stream the given value
      void operator()(const T &value, AprilIO::StreamInterface *stream) {
        UNUSED_VARIABLE(value);
        UNUSED_VARIABLE(stream);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      }
    };

    /**
     * Template for BinaryCoder implementations, it needs to be specialized
     * depending in the Matrix<T> type T.
     */
    template <typename T>
    struct BinaryCoder {
      // puts to the stream the given value
      void operator()(const T &value, AprilIO::StreamInterface *stream) {
        UNUSED_VARIABLE(value);
        UNUSED_VARIABLE(stream);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      }
    };

    /****************************************************************************/
  
    /**
     * Template for SparseAsciiSizer implementations, it needs to be specialized
     * depending in the SparseMatrix<T> type T.
     */
    template <typename T>
    struct SparseAsciiSizer {
      // returns the number of bytes needed for all matrix data (plus spaces)
      int operator()(const SparseMatrix<T> *mat) {
        UNUSED_VARIABLE(mat);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return -1;
      }
    };

    /**
     * Template for SparseBinarySizer implementations, it needs to be specialized
     * depending in the SparseMatrix<T> type T.
     */
    template <typename T>
    struct SparseBinarySizer {
      // returns the number of bytes needed for all matrix data (plus spaces)
      int operator()(const SparseMatrix<T> *mat) {
        UNUSED_VARIABLE(mat);
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return -1;
      }
    };
  } // namespace MatrixIO
} // namespace basics

#endif // MATRIX_SERIALIZATION_UTILS_H
