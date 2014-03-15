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
#include "clamp.h"
#include "aligned_memory.h"
#include "swap.h"
#include "maxmin.h"
#include "qsort.h"
#include "mmapped_data.h"
#include "unused_variable.h"

// CSC or CSR format explained at MKL "Sparse Matrix Storage Formats":
// http://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-9FCEB1C4-670D-4738-81D2-F378013412B0.htm

/// The SparseMatrix class represents bi-dimensional sparse matrices, stored as
/// CSC or CSR format, using one-based indexing (as in Lua). The structure of
/// the matrices is resized dynamically. It is not possible to share internal
/// memory pointers between different matrices, unless transposition operator
/// which shared the data pointers.
template <typename T>
class SparseMatrix : public Referenced {
public:
  enum SPARSE_FORMAT { CSC_FORMAT=0, CSR_FORMAT };
private:
  const static unsigned int MATRIX_BINARY_VERSION;
  const static unsigned int INITIAL_NON_ZERO_SIZE = 8;
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
  /// Number of elements different of zero
  int non_zero_size;
  // Pointers to data: values,indices,ptrB,ptrE
  GPUMirroredMemoryBlock<T> *values;      ///< non-zero values
  IntGPUMirroredMemoryBlock *indices;     ///< indices for rows (CSC) or columns (CSR)
  IntGPUMirroredMemoryBlock *first_index; ///< size(values) + 1
  IntGPUMirroredMemoryBlock *ptrB; ///< first indices (points to first_index)
  IntGPUMirroredMemoryBlock *ptrE; ///< last indices (points to first_index+1)
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
  int searchIndexOf(const int c1, const int c2);
  
public:
  
  /********* Iterators for Matrix traversal *********/
  // forward declaration
  class const_iterator;
  class iterator {
    friend class const_iterator;
    friend class SparseMatrix;
    SparseMatrix<T> *m;
    int idx;
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
    void getCoords(int &x, int &y);
  };
  /*******************************************************/
  class const_iterator {
    friend class SparseMatrix;
    const SparseMatrix<T> *m;
    int idx;
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
    void getCoords(int &x, int &y);
  };
  /*******************************************************/
  
private:
  // const version of iterators, for fast end() iterator calls. They are
  // allocated on-demand, so if end() methods are never executed, they
  // don't waste memory space
  mutable iterator end_iterator;
  mutable const_iterator end_const_iterator;

public:
  /********** Constructors ***********/
  
  /// Constructor
  SparseMatrix(const int d1, const int d2,
	       const SPARSE_FORMAT sparse_format = CSC_FORMAT);
  
  /// Constructor given other matrix, it does a deep copy (clone).
  SparseMatrix(const SparseMatrix<T> *other);
  /// Sub-matrix constructor, makes a deep copy of the given matrix slice
  SparseMatrix(const SparseMatrix<T> *other,
	       const int coord1, const int coord2,
	       const int size1, const int size2);
  /// Destructor
  virtual ~SparseMatrix();
  
  /// Constructor from a MMAP file
  static SparseMatrix<T> *fromMMappedDataReader(april_utils::MMappedDataReader
						*mmapped_data);
  /// Writes to a file
  void toMMappedDataWriter(april_utils::MMappedDataWriter *mmapped_data) const;

  /// Modify sizes of matrix, and returns a cloned sparse matrix
  SparseMatrix<T> *rewrap(const int new_d1, const int new_d2);
  
  /* Getters and setters */
  int getNumDim() const { return 2; }
  const int *getDimPtr() const { return matrixSize; }
  int getDimSize(int i) const { return matrixSize[i]; }
  int size() const { return total_size; }
  int nonZeroSize() const { return non_zero_size; }
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
  iterator iteratorAt(int c1, int c2) {
    int idx = searchIndexOf(c1, c2);
    return iterator(this, idx);
  }
  const iterator &end() {
    if (end_iterator.m == 0)
      end_iterator = iterator(this, non_zero_size+1);
    return end_iterator;
  }
  /************************/
  const_iterator begin() const { return const_iterator(this); }
  const_iterator iteratorAt(int c1, int c2) const {
    int idx = searchIndexOf(c1, c2);
    return iterator(this, idx);
  }
  const const_iterator &end() const {
    if (end_const_iterator.m == 0)
      end_const_iterator = const_iterator(this,
					  non_zero_size+1); 
    return end_const_iterator;
  }

  /// Symbolic transposition, changes the flag and the sparse format
  SparseMatrix<T>* transpose();
  /// Copy only sizes, but not data
  SparseMatrix<T>* cloneOnlyDims() const;
  /// Deep copy
  Matrix<T>* clone();
  /// Deep copy with different sparse format
  Matrix<T> *clone(SPARSE_FORMAT sparse_format);
  
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
  
  /// Adds a new element, with the pre-condition of being added in order. If the
  /// order is not mantained, an error will be thrown
  void pushBack(int c1, int c2, T value);
  
  /// Raw access operator [], access by index position
  T& operator[] (int i);
  const T& operator[] (int i) const;
  /// Access to independent elements
  // T& operator() (int row, int col);
  // const T& operator() (int row, int col) const;
  
  /// Function to obtain RAW access to data pointer. Be careful with it, because
  /// you are losing sub-matrix abstraction, and the major order.
  GPUMirroredMemoryBlock<T> *getRawValuesAccess() { return data; }
  IntGPUMirroredMemoryBlock *getRawIndicesAccess() { return indices; }
  IntGPUMirroredMemoryBlock *getRawPtrBAccess() { return ptrB; }
  IntGPUMirroredMemoryBlock *getRawPtrEAccess() { return ptrE; }

  const GPUMirroredMemoryBlock<T> *getRawValuesAccess() const { return data; }
  const IntGPUMirroredMemoryBlock *getRawIndicesAccess() const { return indices; }
  const IntGPUMirroredMemoryBlock *getRawPtrBAccess() const { return ptrB; }
  const IntGPUMirroredMemoryBlock *getRawPtrEAccess() const { return ptrE; }
  
  /// Returns true if they have the same dimension
  template<typename O>
  bool sameDim(const Matrix<O> *other) const;
  bool sameDim(const int d1, const int d2) const;
  
  ////////////////////////////////////////////////////////////////////////////
  
  void clamp(T lower, T upper);
  // Set a diagonal matrix (only works with a new fresh matrix)
  void diag(T value);

  // Returns a new matrix with the sum, assuming they have the same dimension
  // Crashes otherwise
  SparseMatrix<T>* addition(const SparseMatrix<T> *other);

  // The same as addition but substracting
  SparseMatrix<T>* substraction(const SparseMatrix<T> *other);
  
  // Matrices must be NxK and KxM, the result is NxM
  SparseMatrix<T>* multiply(const SparseMatrix<T> *other) const;

  T sum() const;

  // the argument indicates over which dimension the sum must be performed
  Matrix<T>* sum(int dim, Matrix<T> *dest=0);

  /**** COMPONENT WISE OPERATIONS ****/
  bool equals(const SparseMatrix<T> *other, float epsilon) const;
  void plogp();
  void log();
  void log1p();
  void exp();
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
  void cos();
  void cosh();
  void acos();
  void acosh();
  void abs();
  void complement();
  void sign();
  SparseMatrix<T> *cmul(const SparseMatrix<T> *other);
  void adjustRange(T rmin, T rmax);
  
  /**** BLAS OPERATIONS ****/
  void scalarAdd(T s);
  
  // SCOPY BLAS operation this = other
  void copy(const SparseMatrix<T> *other);
  
  // DOT BLAS operation value = dot(this, other)
  T dot(const SparseMatrix<T> *other) const;
  
  void scal(T value);

  void div(T value);
  
  float norm2() const;
  T min(int &arg_min, int &arg_min_raw_pos) const;
  T max(int &arg_max, int &arg_max_raw_pos) const;
  void minAndMax(T &min, T &max) const;
  
  // Min and max over given dimension, be careful, argmin and argmax matrices
  // contains the min/max index at the given dimension, but starting in 1 (not
  // in 0)
  Matrix<T> *min(int dim, Matrix<T> *dest=0, Matrix<int32_t> *argmin=0);
  Matrix<T> *max(int dim, Matrix<T> *dest=0, Matrix<int32_t> *argmax=0);
  
  // Expands current matrix to a diagonal matrix
  static SparseMatrix<T> *diagonalize(Matrix<T> *diag);
  
private:
  void allocate_memory(int size);
  void release_memory();
  void initialize(int d1, int d2);
};

#include "sparse_matrix.impl.h"
#include "sparse_matrix-iterators.impl.h"
#include "sparse_matrix-math.impl.h"

#endif // SPARSE_MATRIX_H
