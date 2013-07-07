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
#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <cassert>
#include <cstdarg>
#include <new> // surprisingly, placement new doesn't work without this
#include "cblas_headers.h"
#include "wrapper.h"
#include "gpu_mirrored_memory_block.h"
#include "referenced.h"
#include "constants.h"
#include "clamp.h"
#include "aligned_memory.h"
#include "swap.h"
#include "maxmin.h"
#include "qsort.h"

template <typename T>
class Matrix : public Referenced {
protected:
  /// Number of dimensions
  int numDim;
  /// Size of each dimension
  int *stride;
  /// Offset for sub-matrix
  int offset;
  /// Sub-matrix size of each dimension
  int *matrixSize;
  /// Total size of the matrix (number of elements)
  int total_size;
  int last_raw_pos;
  /// Pointer to data
  GPUMirroredMemoryBlock<T> *data;
  /// Major type (only when numDim=2)
  CBLAS_ORDER major_order;
  /// For CUDA purposes
  bool use_cuda;
  
  /// Constructor... -> Integer array with the size of each dimension
  /*
    Matrix(int numDim, const int* dim, T* data_vector,
    CBLAS_ORDER major_order = CblasRowMajor);
  */
  /// Computes the position at data array given it coordinates
  int  computeRawPos(const int *coords) const;
  /// Computes the coordinates given the raw data position
  void computeCoords(const int raw_pos, int *coords) const;
  /// Returns the data pointer for read and write
  T *getData() { return data->getPPALForReadAndWrite(); }
  /// Returns the data pointer for read
  const T *getData() const { return data->getPPALForRead(); }
  
  int getLastRawPos() const { return last_raw_pos; }
  /// Returns if the matrix is a vector
  bool isVector() const { return (numDim==1 ||
				  (numDim==2 &&
				   ( (matrixSize[0]==1) ||
				     (matrixSize[1]==1) ))); }
  bool isColVector() const { return (numDim==2 && matrixSize[1]==1); }
  /// Returns the size of the vector, the coordinate which is different of 1. It
  /// only works if the matrix is a vector (precondition).
  int getVectorSize() const {
    return ( (numDim==1) ? matrixSize[0] :
	     april_utils::max(matrixSize[0], matrixSize[1]) ); }
  int getVectorStride() const {
    return (numDim == 1) ? stride[0] :
      (major_order==CblasRowMajor) ? stride[1] : stride[0];
  }

  void scalarAdd(T s);
  
public:
  /// Updates with the following coordinates vector
  bool nextCoordVectorRowOrder(int *coords, int &raw_pos) const;
  bool nextCoordVectorColOrder(int *coords, int &raw_pos) const;
  bool nextCoordVectorRowOrder(int *coords) const;
  bool nextCoordVectorColOrder(int *coords) const;
  static bool nextCoordVectorRowOrder(int *coords, const int *sizes,
				      const int numDim);
  static bool nextCoordVectorColOrder(int *coords, const int *sizes,
				      const int numDim);
  static bool nextCoordVectorRowOrder(int *coords, int &raw_pos,
				      const int *sizes,
				      const int *strides,
				      const int numDim,
				      const int last_raw_pos);
  static bool nextCoordVectorColOrder(int *coords, int &raw_pos,
				      const int *sizes,
				      const int *strides,
				      const int numDim,
				      const int last_raw_pos);
  /********* Iterators for Matrix traversal *********/
  // forward declaration
  class const_iterator;
  class col_major_iterator;
  class const_col_major_iterator;
  class iterator {
    friend class const_iterator;
    friend class col_major_iterator;
    friend class const_col_major_iterator;
    friend class Matrix;
    Matrix *m;
    int idx;
    int raw_pos;
    /// The coords array is only used when the matrix is not congiuous
    /// or it is in col_major order
    int *coords;
    T *data;
    iterator(Matrix *m);
    iterator(Matrix *m, int raw_pos);
    iterator(Matrix *m, int raw_pos, int *coords);
  public:
    iterator();
    iterator(const iterator &other);
    ~iterator();
    iterator &operator=(const iterator &other);
    bool      operator==(const iterator &other) const;
    bool      operator!=(const iterator &other) const;
    iterator &operator++();
    T &operator*();
    int getRawPos() const;
    int getIdx() const { return idx; }
  };
  /*******************************************************/
  class col_major_iterator {
    friend class Matrix;
    Matrix *m;
    int idx;
    int raw_pos;
    /// The coords array is only used when the matrix is not congiuous
    /// or it is in row_major order
    int *coords;
    T *data;
    col_major_iterator(Matrix *m);
    col_major_iterator(Matrix *m, int raw_pos);
    col_major_iterator(Matrix *m, int raw_pos, int *coords);
  public:
    col_major_iterator();
    col_major_iterator(const col_major_iterator &other);
    col_major_iterator(const iterator &other);
    ~col_major_iterator();
    col_major_iterator &operator=(const col_major_iterator &other);
    col_major_iterator &operator=(const iterator &other);
    bool      operator==(const col_major_iterator &other) const;
    bool      operator==(const iterator &other) const;
    bool      operator!=(const col_major_iterator &other) const;
    bool      operator!=(const iterator &other) const;
    col_major_iterator &operator++();
    T &operator*();
    int getRawPos() const;
    int getIdx() const { return idx; }
  };
  /*******************************************************/
  class const_iterator {
    friend class const_col_major_iterator;
    friend class Matrix;
    const Matrix *m;
    int idx;
    int raw_pos;
    /// The coords array is only used when the matrix is not congiuous
    /// or it is in col_major order
    int *coords;
    const T *data;
    const_iterator(const Matrix *m);
    const_iterator(const Matrix *m, int raw_pos);
    const_iterator(const Matrix *m, int raw_pos, int *coords);
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
    int getRawPos() const;
    int getIdx() const { return idx; }
  };
  /*******************************************************/
  class const_col_major_iterator {
    friend class Matrix;
    const Matrix *m;
    int idx;
    int raw_pos;
    /// The coords array is only used when the matrix is not congiuous
    /// or it is in row_major order
    int *coords;
    const T *data;
    const_col_major_iterator(const Matrix *m);
    const_col_major_iterator(const Matrix *m, int raw_pos);
    const_col_major_iterator(const Matrix *m, int raw_pos, int *coords);
  public:
    const_col_major_iterator();
    const_col_major_iterator(const const_col_major_iterator &other);
    const_col_major_iterator(const iterator &other);
    const_col_major_iterator(const const_iterator &other);
    /*const_col_major_iterator(const iterator &other);*/
    ~const_col_major_iterator();
    const_col_major_iterator &operator=(const const_col_major_iterator &other);
    const_col_major_iterator &operator=(const iterator &other);
    const_col_major_iterator &operator=(const const_iterator &other);
    bool            operator==(const const_col_major_iterator &other) const;
    bool            operator==(const iterator &other) const;
    bool            operator==(const const_iterator &other) const;
    bool            operator!=(const const_col_major_iterator &other) const;
    bool            operator!=(const iterator &other) const;
    bool            operator!=(const const_iterator &other) const;
    const_col_major_iterator &operator++();
    const T &operator*() const;
    int getRawPos() const;
    int getIdx() const { return idx; }
  };
  
private:
  /********************************************************/
  // The span iterator traverses the matrix allowing to do a linear
  // traversal of the largest dimension.
  // ATTENTION: Currently it is a private iterator
  class best_span_iterator {
    friend class Matrix;
    Matrix *m;
    int raw_pos;
    int *coords, *order;
    struct inverse_sort_compare {
      Matrix *m;
      inverse_sort_compare(Matrix *m) : m(m) { }
      bool operator()(const int &a, const int &b) {
	// FIXME: Would be better to use a trade-off between size and stride?
	return m->matrixSize[a] > m->matrixSize[b];
      }
    };
    best_span_iterator(Matrix *m, int raw_pos) : m(m), raw_pos(raw_pos) {
      coords = new int[m->numDim];
      order  = new int[m->numDim];
      m->computeCoords(raw_pos, coords);
      switch(m->numDim) {
      case 1: order[0] = 0; break;
      case 2:
	if (m->matrixSize[0] > m->matrixSize[1]) {
	  order[0] = 0;
	  order[1] = 1;
	}
	else {
	  order[0] = 1;
	  order[1] = 0;
	}
	break;
      default:
	for (int i=0; i<m->numDim; ++i) order[i] = i;
	april_utils::Sort(order, 0, m->numDim-1, inverse_sort_compare(m));
      }
    }
    best_span_iterator(Matrix *m) : m(m), raw_pos(m->offset) {
      coords = new int[m->numDim];
      order  = new int[m->numDim];
      switch(m->numDim) {
      case 1: order[0] = 0; coords[0] = 0; break;
      case 2:
	coords[0] = 0; coords[1] = 0;
	if (m->matrixSize[0] > m->matrixSize[1]) {
	  order[0] = 0;
	  order[1] = 1;
	}
	else {
	  order[0] = 1;
	  order[1] = 0;
	}
	break;
      default:
	for (int i=0; i<m->numDim; ++i) {
	  coords[i] = 0;
	  order[i] = i;
	}
	april_utils::Sort(order, 0, m->numDim-1, inverse_sort_compare(m));
      }
    }
  public:
    best_span_iterator() : m(0), coords(0), order(0) { }
    ~best_span_iterator() {
      delete[] order;
      delete[] coords;
    }
    int getOffset() const { return raw_pos; }
    int getStride() const { return m->stride[order[0]]; }
    int getSize() const { return m->matrixSize[order[0]]; }
    best_span_iterator &operator=(const best_span_iterator &other) {
      if (m==0 || m->numDim != other.m->numDim) {
	delete[] coords;
	delete[] order;
	coords = new int[other.m->getNumDim()];
	order = new int[other.m->getNumDim()];
      }
      m = other.m;
      raw_pos = other.raw_pos;
      for (int i=0; i<m->getNumDim(); ++i) {
	order[i] = other.order[i];
	coords[i] = other.coords[i];
      }
      return *this;
    }
    bool      operator==(const best_span_iterator &other) const {
      return m==other.m && raw_pos==other.raw_pos;
    }
    bool      operator!=(const best_span_iterator &other) const {
      return !((*this)==other);
    }
    best_span_iterator &operator++() {
      switch(m->numDim) {
      case 1: raw_pos = m->last_raw_pos+1; break;
      case 2:
	coords[1] = (coords[1]+1) % m->matrixSize[order[1]];
	if (coords[1] > 0) raw_pos += m->stride[order[1]];
	else raw_pos = m->last_raw_pos+1;
	break;
      default:
	int j = 1, pos;
	do {
	  pos = order[j++];
	  coords[pos] = (coords[pos]+1) % m->matrixSize[pos];
	} while(j<m->numDim && coords[pos] == 0);
	if (j == m->numDim && coords[pos] == 0) raw_pos = m->last_raw_pos+1;
	else raw_pos = m->computeRawPos(coords);
      }
      return *this;
    }
  };

  // const version of iterators, for fast end() iterator calls. They are
  // allocated on-demand, so if end() methods are never executed, they
  // don't waste memory space
  const iterator end_iterator;
  const const_iterator end_const_iterator;
  const best_span_iterator end_best_span_iterator;
  
  const best_span_iterator &end_span_iterator() {
    if (end_best_span_iterator.m == 0)
      *(const_cast<best_span_iterator*>(&end_best_span_iterator)) =
	best_span_iterator(this, last_raw_pos+1);
    return end_best_span_iterator;
  }
  
public:
  /********** Constructors ***********/
  /// Full constructor given numDim, dim, and major_order
  Matrix(int numDim, const int* dim,
	 CBLAS_ORDER major_order = CblasRowMajor,
	 GPUMirroredMemoryBlock<T> *data = 0,
	 int offset = 0);
  
  /// Constructor with T() values and CblasRowMajor order
  Matrix(int numDim, int d1, ...);
  
  /// Constructor given other matrix, it does a shallow or deep copy (clone). By
  /// default is a deep copy, some code pieces expect this behavior, don't
  /// change it.
  Matrix(Matrix<T> *other, bool clone=true);
  /// Sub-matrix constructor
  Matrix(Matrix<T> *other,
	 const int* coords, const int *sizes,
	 bool clone=true);
  /// Destructor
  virtual ~Matrix();
  
  /// Modify sizes of matrix
  Matrix<T> *rewrap(const int *new_dims, int len);
  
  /* Getters and setters */
  int getNumDim() const { return numDim; }
  const int *getDimPtr() const { return matrixSize; }
  const int *getStridePtr() const { return stride; }
  int getDimSize(int i) const { return matrixSize[i]; }
  int getStrideSize(int i) const { return stride[i]; }
  int size() const { return total_size; }
  CBLAS_ORDER getMajorOrder() const { return major_order; }
  void setUseCuda(bool v) { use_cuda = v; }
  bool getCudaFlag() const { return use_cuda; }
  bool isSimple() const {
    return (getIsContiguous())&&(major_order==CblasRowMajor);
  }
  /// Indicates if it is a contiguous matrix
  bool getIsContiguous() const;
  /// Returns the offset of first data value (sub-matrix)
  int getOffset() const { return offset; }
  /**********************/
  iterator begin() { return iterator(this); }
  iterator iteratorAt(int c0) {
    assert(numDim==1);
    return iterator(this, computeRawPos(&c0), &c0);
  }
  iterator iteratorAt(int c0, int c1) {
    assert(numDim==2);
    int aux[2]={c0,c1};
    return iterator(this, computeRawPos(aux), aux);
  }
  iterator iteratorAt(int *coords, int len) {
    assert(numDim==len);
    return iterator(this, computeRawPos(coords), coords);
  }
  const iterator &end() {
    if (end_iterator.m == 0)
      *(const_cast<iterator*>(&end_iterator)) = iterator(this, last_raw_pos+1);
    return end_iterator;
  }
  /************************/
  const_iterator begin() const { return const_iterator(this); }
  const_iterator iteratorAt(int c0) const {
    assert(numDim==1);
    return const_iterator(this, computeRawPos(&c0), &c0);
  }
  const_iterator iteratorAt(int c0, int c1) const {
    assert(numDim==2);
    int aux[2]={c0,c1};
    return const_iterator(this, computeRawPos(aux), aux);
  }
  const_iterator iteratorAt(int *coords, int len) const {
    assert(numDim==len);
    return const_iterator(this, computeRawPos(coords), coords);
  }
  const const_iterator &end() const {
    if (end_const_iterator.m == 0)
      *(const_cast<const_iterator*>(&end_const_iterator)) =
	const_iterator(this,
		       last_raw_pos+1); 
    return end_const_iterator;
  }

  /// Transposition
  Matrix<T>* transpose() const;
  /// Copy only sizes, but not data
  Matrix<T>* cloneOnlyDims() const;
  /// Deep copy
  Matrix<T>* clone();
  /// Deep copy with different major_order
  Matrix<T> *clone(CBLAS_ORDER major_order);
  /// Shallow copy
  Matrix<T>* shallow_copy();
  /// Raw access operator []
  T& operator[] (int i);
  const T& operator[] (int i) const;
  // Access to independent elements, one and two dimensions are special cases
  T& operator() (int i);
  T& operator() (int row, int col);
  T& operator() (int coord0, int coord1, int coord2, ...);
  T& operator() (int *coords, int sz);
  const T& operator() (int i) const;
  const T& operator() (int row, int col) const;
  const T& operator() (int coord0, int coord1, int coord2, ...) const;
  const T& operator() (int *coords, int sz) const;
  
  /// Function to obtain RAW access to data pointer. Be careful with it, because
  /// you are losing sub-matrix abstraction, and the major order.
  GPUMirroredMemoryBlock<T> *getRawDataAccess() { return data; }
  const GPUMirroredMemoryBlock<T> *getRawDataAccess() const { return data; }
  
  bool getCol(int col, T* vec, int vecsize);
  bool putCol(int col, T *vec, int vecsize);
  bool putSubCol(int col, int first_row, T *vec, int vecsize);

  // Returns true if they have the same dimension
  bool sameDim(const Matrix *other) const;
  
  ////////////////////////////////////////////////////////////////////////////

  void clamp(T lower, T upper);
  void fill(T value);
  void zeros();
  void ones();
  
  // Set the diagonal to current value
  void diag(T value);

  // Returns a new matrix with the sum, assuming they have the same dimension
  // Crashes otherwise
  Matrix<T>* addition(const Matrix<T> *other);

  // The same as addition but substracting
  Matrix<T>* substraction(const Matrix<T> *other);
  
  // Matrices must be NxK and KxM, the result is NxM
  Matrix<T>* multiply(const Matrix<T> *other) const;

  T sum() const;
  
  /**** COMPONENT WISE OPERATIONS ****/
  bool equals(const Matrix<T> *other, T epsilon) const;
  void log();
  void log1p();
  void exp();
  void sqrt();
  void pow(T value);
  void tanh();
  Matrix<T> *cmul(const Matrix<T> *other);
  void adjustRange(T rmin, T rmax);
    
  /**** BLAS OPERATIONS ****/
  
  // FIXME: This operations could be improved if we take into account when the
  // matrix data is contiguous in memory (even when it is a sub-matrix)

  // SCOPY BLAS operation this = other
  void copy(const Matrix<T> *other);
  
  // AXPY BLAS operation this = this + alpha * other
  void axpy(T alpha, const Matrix<T> *other);
  
  // GEMM BLAS operation this = alpha * op(A)*op(B) + beta*this
  void gemm(CBLAS_TRANSPOSE trans_A,
	    CBLAS_TRANSPOSE trans_B,
	    T alpha,
	    const Matrix<T> *otherA,
	    const Matrix<T> *otherB,
	    T beta);

  // GEMV BLAS operation this = alpha * op(A)*X + beta*this
  void gemv(CBLAS_TRANSPOSE trans_A,
	    T alpha,
	    const Matrix<T> *otherA,
	    const Matrix<T> *otherX,
	    T beta);

  // GER BLAS operation this = alpha * X*Y' + this
  void ger(T alpha,
	   const Matrix<T> *otherX,
	   const Matrix<T> *otherY);

  // DOT BLAS operation value = dot(this, other)
  T dot(const Matrix<T> *other) const;
  
  void scal(T value);
  
  T norm2() const;
  T min(int &arg_min) const;
  T max(int &arg_max) const;
  void minAndMax(T &min, T &max) const;
  
private:
  void allocate_memory(int size);
  void release_memory();
  void initialize(const int *dim);
};

#include "matrix.impl.h"
#include "matrix-iterators.impl.h"
#include "matrix-math.impl.h"

#endif // MATRIX_H
