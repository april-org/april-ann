/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <cmath>
#include "april_assert.h"
#include <cstdarg>
#include <new> // surprisingly, placement new doesn't work without this
#include "cblas_headers.h"
#include "wrapper.h"
#include "gpu_mirrored_memory_block.h"
#include "referenced.h"
#include "aligned_memory.h"
#include "swap.h"
#include "maxmin.h"
#include "qsort.h"
#include "mmapped_data.h"
#include "unused_variable.h"

// CSC or CSR format explained at MKL "Sparse Matrix Storage Formats":
// http://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-9FCEB1C4-670D-4738-81D2-F378013412B0.htm

/// The SparseMatrix class represents bi-dimensional sparse matrices, stored as
/// CSC or CSR format, using zero-based indexing (as in Lua). The structure of
/// the matrices is resized dynamically. It is not possible to share internal
/// memory pointers between different matrices, unless transposition operator
/// which shared the data pointers.
template <typename T>
class SparseMatrix : public Referenced {
  const static unsigned int MATRIX_BINARY_VERSION;
  // Auxiliary count variable where the user could store the number of times
  // this object is shared in a computation (like in ANN components sharing
  // weight matrices)
  unsigned int shared_count;
protected:
  /// Number of dimensions (always 2)
  static const int numDim = 2;
  /// size of each dimension
  int matrixSize[2];
  /// Total size of the matrix (number of elements)
  int total_size;
  // Pointers to data: values,indices,first_index
  GPUMirroredMemoryBlock<T> *values;      ///< non-zero values
  Int32GPUMirroredMemoryBlock *indices;     ///< indices for rows (CSC) or columns (CSR)
  Int32GPUMirroredMemoryBlock *first_index; ///< size(values) + 1
  /// For mmapped matrices
  april_utils::MMappedDataReader *mmapped_data;
  /// Format type (CSC or CSR)
  SPARSE_FORMAT sparse_format;
  /// For CUDA purposes
  bool use_cuda;
  
  /// Constructor... -> Integer array with the size of each dimension
  /*
    Matrix(int numDim, const int* dim, T* data_vector,
    CBLAS_ORDER major_order = CblasRowMajor);
  */
  
  /// Returns if the matrix is a vector
  bool isVector() const { return ( (matrixSize[0]==1) ||
				   (matrixSize[1]==1) ); }
  bool isColVector() const { return matrixSize[1]==1; }
  int searchIndexOf(const int c0, const int c1) const;
  int searchIndexOfFirst(const int c0, const int c1) const;
  
public:
  
  /********* Iterators for Matrix traversal *********/
  // forward declaration
  class const_iterator;
  class iterator {
    friend class const_iterator;
    friend class SparseMatrix;
    SparseMatrix<T> *m;
    int idx;
    //
    T   *values;
    int *indices;
    int *first_index;
    int  first_index_pos;
    iterator(SparseMatrix<T> *m, int idx=0);
  public:
    iterator();
    iterator(const iterator &other);
    ~iterator();
    iterator &operator=(const iterator &other);
    bool      operator==(const iterator &other) const;
    bool      operator!=(const iterator &other) const;
    iterator &operator++();
    T &operator*();
    T *operator->();
    int getIdx() const { return idx; }
    void getCoords(int &x0, int &x1) const;
  };
  /*******************************************************/
  class const_iterator {
    friend class SparseMatrix;
    const SparseMatrix<T> *m;
    int idx;
    //
    const T   *values;
    const int *indices;
    const int *first_index;
    int  first_index_pos;
    const_iterator(const SparseMatrix<T> *m, int idx=0);
  public:
    const_iterator();
    const_iterator(const const_iterator &other);
    const_iterator(const iterator &other);
    /*const_iterator(const iterator &other);*/
    ~const_iterator();
    const_iterator &operator=(const const_iterator &other);
    const_iterator &operator=(const iterator &other);
    bool            operator==(const const_iterator &other) const;
    bool            operator==(const iterator &other) const;
    bool            operator!=(const const_iterator &other) const;
    bool            operator!=(const iterator &other) const;
    const_iterator &operator++();
    const T &operator*() const;
    const T *operator->() const;
    int getIdx() const { return idx; }
    void getCoords(int &x0, int &x1) const;
  };
  /*******************************************************/
  
private:
  // const version of iterators, for fast end() iterator calls. They are
  // allocated on-demand, so if end() methods are never executed, they
  // nerver will be initialized.
  mutable iterator end_iterator;
  mutable const_iterator end_const_iterator;
  
  SparseMatrix() : end_iterator(), end_const_iterator() { }
  void checkSortedIndices(bool sort=false);
  
public:
  /********** Constructors ***********/
  
  /// Constructor
  SparseMatrix(const int d0, const int d1,
	       GPUMirroredMemoryBlock<T> *values,
	       Int32GPUMirroredMemoryBlock *indices,
	       Int32GPUMirroredMemoryBlock *first_index,
	       const SPARSE_FORMAT sparse_format = CSR_FORMAT,
	       bool sort=false);

  /// Constructor given a dense matrix, it does constructs a sparse matrix
  /// (cloned).
  SparseMatrix(const Matrix<T> *other,
	       const SPARSE_FORMAT sparse_format = CSR_FORMAT);
  /// Constructor given other matrix, it does a deep copy (clone).
  SparseMatrix(const SparseMatrix<T> *other,
	       SPARSE_FORMAT sparse_format = NONE_FORMAT);
  /// Sub-matrix constructor, makes a deep copy of the given matrix slice
  SparseMatrix(const SparseMatrix<T> *other,
	       const int *coords, const int *sizes, bool clone=true);
  /// Destructor
  virtual ~SparseMatrix();
  
  /// Constructor from a MMAP file
  static SparseMatrix<T> *fromMMappedDataReader(april_utils::MMappedDataReader
						*mmapped_data);
  /// Writes to a file
  void toMMappedDataWriter(april_utils::MMappedDataWriter *mmapped_data) const;

  /* Getters and setters */
  int getNumDim() const { return numDim; }
  const int *getDimPtr() const { return matrixSize; }
  int getDimSize(int i) const { return matrixSize[i]; }
  int size() const { return total_size; }
  // FIXME: use an attribute to improve the efficiency of this call
  int nonZeroSize() const { return static_cast<int>(values->getSize()); }
  int getDenseCoordinateSize() const {
    if (sparse_format == CSR_FORMAT) return matrixSize[0];
    else return matrixSize[1];
  }
  int getCompressedCoordinateSize() const {
    if (sparse_format == CSR_FORMAT) return matrixSize[1];
    else return matrixSize[0];
  }
  SPARSE_FORMAT getSparseFormat() const { return sparse_format; }
  bool getIsDataRowOrdered() const {
    return sparse_format == CSR_FORMAT;
  }
  void setUseCuda(bool v) {
    use_cuda = v;
#ifdef USE_CUDA
    if (use_cuda) {
      values->updateMemGPU();
      indices->updateMemGPU();
      first_index->updateMemGPU();
    }
#endif
  }
  bool getCudaFlag() const { return use_cuda; }
  /**********************/
  iterator begin() { return iterator(this); }
  iterator iteratorAt(int c0, int c1) {
    int idx = searchIndexOf(c0, c1);
    if (idx == -1) ERROR_EXIT2(128,"Incorrect given position (%d,%d)\n",c0,c1);
    return iterator(this, idx);
  }
  iterator iteratorAtFirst(int c0, int c1) {
    int idx = searchIndexOfFirst(c0, c1);
    return iterator(this, idx);
  }
  iterator iteratorAtRawIndex(int i) {
    return iterator(this,i);
  }
  const iterator &end() {
    if (end_iterator.m == 0)
      end_iterator = iterator(this, nonZeroSize());
    return end_iterator;
  }
  /************************/
  const_iterator begin() const { return const_iterator(this); }
  const_iterator iteratorAt(int c0, int c1) const {
    int idx = searchIndexOf(c0, c1);
    if (idx == -1) ERROR_EXIT2(128,"Incorrect given position (%d,%d)\n",c0,c1);
    return const_iterator(this, idx);
  }
  const_iterator iteratorAtFirst(int c0, int c1) const {
    int idx = searchIndexOfFirst(c0, c1);
    return const_iterator(this, idx);
  }
  const_iterator iteratorAtRawIndex(int i) const {
    return const_iterator(this,i);
  }
  const const_iterator &end() const {
    if (end_const_iterator.m == 0)
      end_const_iterator = const_iterator(this,
					  nonZeroSize()); 
    return end_const_iterator;
  }

  /// Symbolic transposition, changes the sparse format
  SparseMatrix<T>* transpose();
  /// Deep copy with different sparse format
  SparseMatrix<T> *clone(SPARSE_FORMAT sparse_format = NONE_FORMAT) const;
  
  /// Returns an equivalent dense matrix
  Matrix<T> *toDense(CBLAS_ORDER order=CblasRowMajor) const;
  
  /// Number values check
  void pruneSubnormalAndCheckNormal();
  
  /// This method sets to zero the shared counter
  void resetSharedCount() { shared_count = 0; }
  /// This method adds counts to the shared counter
  void addToSharedCount(unsigned int count=1) { shared_count += count; }
  /// Getter of the shared count value
  unsigned int getSharedCount() const {
    return shared_count;
  }
  
  /// Raw access operator [], access by index position
  T& operator[] (int i);
  const T& operator[] (int i) const;
  /// Access to independent elements
  // T& operator() (int row, int col);
  const T operator() (int row, int col) const;
  
  /// Function to obtain RAW access to data pointer. Be careful with it, because
  /// you are losing sub-matrix abstraction, and the major order.
  GPUMirroredMemoryBlock<T> *getRawValuesAccess() { return values; }
  Int32GPUMirroredMemoryBlock *getRawIndicesAccess() { return indices; }
  Int32GPUMirroredMemoryBlock *getRawFirstIndexAccess() { return first_index; }

  const GPUMirroredMemoryBlock<T> *getRawValuesAccess() const { return values; }
  const Int32GPUMirroredMemoryBlock *getRawIndicesAccess() const { return indices; }
  const Int32GPUMirroredMemoryBlock *getRawFirstIndexAccess() const { return first_index; }
  
  /// Returns true if they have the same dimension
  template<typename O>
  bool sameDim(const SparseMatrix<O> *other) const;
  template<typename O>
  bool sameDim(const Matrix<O> *other) const;
  bool sameDim(const int d0, const int d1) const;
  
  ////////////////////////////////////////////////////////////////////////////

  bool isDiagonal() const {
    if (matrixSize[0] != matrixSize[1] || matrixSize[0] != nonZeroSize())
      return false;
    for(const_iterator it(begin()); it!=end(); ++it) {
      int c0,c1;
      it.getCoords(c0,c1);
      if (c0 != c1) return false;
    }
    return true;
  }
  
  void fill(T value);
  void zeros();
  void ones();
  static SparseMatrix<T> *diag(int N, T value=T(),
			       SPARSE_FORMAT sparse_format = CSR_FORMAT) {
    unsigned int uN = static_cast<unsigned int>(N);
    SparseMatrix<T> *result;
    GPUMirroredMemoryBlock<T> *values = new GPUMirroredMemoryBlock<T>(uN);
    Int32GPUMirroredMemoryBlock *indices = new Int32GPUMirroredMemoryBlock(uN);
    Int32GPUMirroredMemoryBlock *first_index = new Int32GPUMirroredMemoryBlock(uN+1);
    T *values_ptr = values->getPPALForWrite();
    int *indices_ptr = indices->getPPALForWrite();
    int *first_index_ptr = first_index->getPPALForWrite();
    first_index_ptr[0] = 0;
    for (unsigned int i=0; i<uN; ++i) {
      values_ptr[i] = value;
      indices_ptr[i] = i;
      first_index_ptr[i+1] = i+1;
    }
    result = new SparseMatrix<T>(N, N,
				 values, indices, first_index,
				 sparse_format);
    return result;
  }
  
  static SparseMatrix<T> *diag(Matrix<T> *m,
			       SPARSE_FORMAT sparse_format = CSR_FORMAT) {
    if (m->getNumDim() != 1)
      ERROR_EXIT(128, "Expected uni-dimensional matrix\n");
    int N = m->getDimSize(0);
    unsigned int uN = static_cast<unsigned int>(N);
    SparseMatrix<T> *result;
    GPUMirroredMemoryBlock<T> *values = new GPUMirroredMemoryBlock<T>(uN);
    Int32GPUMirroredMemoryBlock *indices = new Int32GPUMirroredMemoryBlock(uN);
    Int32GPUMirroredMemoryBlock *first_index = new Int32GPUMirroredMemoryBlock(uN+1);
    T *values_ptr = values->getPPALForWrite();
    int *indices_ptr = indices->getPPALForWrite();
    int *first_index_ptr = first_index->getPPALForWrite();
    first_index_ptr[0] = 0;
    typename Matrix<T>::const_iterator it(m->begin());
    for (unsigned int i=0; i<uN; ++i, ++it) {
      april_assert(it != m->end());
      values_ptr[i] = *it;
      indices_ptr[i] = i;
      first_index_ptr[i+1] = i+1;
    }
    result = new SparseMatrix<T>(N, N,
				 values, indices, first_index,
				 sparse_format);
    return result;
  }

  static SparseMatrix<T> *diag(GPUMirroredMemoryBlock<T> *values,
			       SPARSE_FORMAT sparse_format = CSR_FORMAT) {
    unsigned int uN = values->getSize();
    int N = static_cast<int>(uN);
    SparseMatrix<T> *result;
    Int32GPUMirroredMemoryBlock *indices = new Int32GPUMirroredMemoryBlock(uN);
    Int32GPUMirroredMemoryBlock *first_index = new Int32GPUMirroredMemoryBlock(uN+1);
    int *indices_ptr = indices->getPPALForWrite();
    int *first_index_ptr = first_index->getPPALForWrite();
    first_index_ptr[0] = 0;
    for (unsigned int i=0; i<uN; ++i) {
      indices_ptr[i] = i;
      first_index_ptr[i+1] = i+1;
    }
    result = new SparseMatrix<T>(N, N,
				 values, indices, first_index,
				 sparse_format);
    return result;
  }

  T sum() const;

  // the argument indicates over which dimension the sum must be performed
  Matrix<T>* sum(int dim, Matrix<T> *dest=0);

  /**** COMPONENT WISE OPERATIONS ****/
  bool equals(const SparseMatrix<T> *other, float epsilon) const;
  void sqrt();
  void pow(T value);
  void tan();
  void tanh();
  void atan();
  void atanh();
  void sin();
  void sinh();
  void asin();
  void asinh();
  void abs();
  void sign();
  
  /**** BLAS OPERATIONS ****/
  
  // SCOPY BLAS operation this = other
  void copy(const SparseMatrix<T> *other);
  
  void scal(T value);

  void div(T value);
  
  float norm2() const;
  T min(int &c0, int &c1) const;
  T max(int &c0, int &c1) const;
  void minAndMax(T &min, T &max) const;
  
  // Min and max over given dimension, be careful, argmin and argmax matrices
  // contains the min/max index at the given dimension, but starting in 1 (not
  // in 0)
  Matrix<T> *min(int dim, Matrix<T> *dest=0, Matrix<int32_t> *argmin=0);
  Matrix<T> *max(int dim, Matrix<T> *dest=0, Matrix<int32_t> *argmax=0);
  
  /// This method converts the caller SparseMatrix in a vector, unrolling the
  /// dimensions in row-major order. If the SparseMatrix is in CSR format, the
  /// resulting vector is a row vector, otherwise, it is a column vector.
  SparseMatrix<T> *asVector() const;  
  
private:
  void allocate_memory(int size);
  void release_memory();
  void initialize(int d0, int d1);
};

#include "sparse_matrix.impl.h"
#include "sparse_matrix-iterators.impl.h"
#include "sparse_matrix-math.impl.h"

#endif // SPARSE_MATRIX_H
