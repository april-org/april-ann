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
#include <cstdarg>
#include <new> // surprisingly, placement new doesn't work without this
#include "aligned_memory.h"
#include "april_assert.h"
#include "c_string.h"
#include "cblas_headers.h"
#include "clamp.h"
#include "disallow_class_methods.h"
#include "gpu_mirrored_memory_block.h"
#include "maxmin.h"
#include "mmapped_data.h"
#include "logbase.h"
#include "qsort.h"
#include "serializable.h"
#include "smart_ptr.h"
#include "swap.h"
#include "unused_variable.h"
#include "mathcore.h"

namespace basics {
  
  namespace MatrixIO {
    /// Boolean option key for read/write using tabulated format.
    const char * const TAB_OPTION   = "tab";
    /// Boolean option key for read/write using ascii format.
    const char * const ASCII_OPTION = "ascii";
    /// String option key for read/write in 'col_major' or 'row_major'.
    const char * const ORDER_OPTION = "order";
    /// String option key with a delimitiers list.
    const char * const DELIM_OPTION = "delim";
    /// Boolean option key indicating if empty fields are allowed during read.
    const char * const EMPTY_OPTION = "read_empty";
    /// T option key indicating the default value for empty fields.
    const char * const DEFAULT_OPTION = "default";
    /// T option key indicating the expected number of columns.
    const char * const NCOLS_OPTION = "ncols";
    /// T option key indicating the expected number of rows.
    const char * const NROWS_OPTION = "nrows";
  }

  // forward declaration
  template <typename T>
  class SparseMatrix;

  /**
   * @brief Multidimensional matrix class.
   * 
   * It implements basic tensor operations.
   */
  template <typename T>
  class Matrix : public AprilIO::Serializable {
    APRIL_DISALLOW_ASSIGN(Matrix);
    //
    friend class SparseMatrix<T>;
    const static unsigned int MATRIX_BINARY_VERSION;
    enum matrix_contiguous_enum_t { NONE=0, CONTIGUOUS=1, NONCONTIGUOUS=2 };
    // Auxiliary count variable where the user could store the number of times
    // this object is shared in a computation (like in ANN components sharing
    // weight matrices)
    unsigned int shared_count;
  protected:
    /// Indicator of transposition
    bool transposed;
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
    april_utils::SharedPtr< april_math::GPUMirroredMemoryBlock<T> > data;
    april_utils::SharedPtr< april_utils::MMappedDataReader > mmapped_data;
    /// Major type (only when numDim=2)
    CBLAS_ORDER major_order;
    /// For CUDA purposes
    bool use_cuda;
    /// To know if it is contiguous
    mutable matrix_contiguous_enum_t is_contiguous;
  
    /// Constructor... -> Integer array with the size of each dimension
    /*
      Matrix(int numDim, const int* dim, T* data_vector,
      CBLAS_ORDER major_order = CblasRowMajor);
    */
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
    /// Returns the stride of the vector, the stride whom coordinate is different
    /// of 1. It only works if the matrix is a vector (precondition).
    int getVectorStride() const {
      return (numDim == 1) ? stride[0] :
        ( (matrixSize[0]!=1) ? (stride[0]) : (stride[1]) );
    }
  
  public:
    class sliding_window;
    friend class sliding_window;
  
    /// Computes the position at data array given it coordinates
    int  computeRawPos(const int *coords) const;
    /// Computes the coordinates given the raw data position
    void computeCoords(const int raw_pos, int *coords) const;
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
      Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous
      /// or it is in col_major order, otherwise it is NULL
      int *coords;
      T *data;
      iterator(Matrix<T> *m);
      iterator(Matrix<T> *m, int raw_pos);
      iterator(Matrix<T> *m, int raw_pos, int *coords);
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
      int getRawPos() const;
      int getIdx() const { return idx; }
    };
    /*******************************************************/
    class col_major_iterator {
      friend class Matrix;
      Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous
      /// or it is in row_major order, otherwise it is NULL
      int *coords;
      T *data;
      col_major_iterator(Matrix<T> *m);
      col_major_iterator(Matrix<T> *m, int raw_pos);
      col_major_iterator(Matrix<T> *m, int raw_pos, int *coords);
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
      T *operator->();
      int getRawPos() const;
      int getIdx() const { return idx; }
    };
    /*******************************************************/
    class const_iterator {
      friend class const_col_major_iterator;
      friend class Matrix;
      const Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous
      /// or it is in col_major order, otherwise it is NULL
      int *coords;
      const T *data;
      const_iterator(const Matrix<T> *m);
      const_iterator(const Matrix<T> *m, int raw_pos);
      const_iterator(const Matrix<T> *m, int raw_pos, int *coords);
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
      int getRawPos() const;
      int getIdx() const { return idx; }
    };
    /*******************************************************/
    class const_col_major_iterator {
      friend class Matrix;
      const Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous
      /// or it is in row_major order, otherwise it is NULL
      int *coords;
      const T *data;
      const_col_major_iterator(const Matrix<T> *m);
      const_col_major_iterator(const Matrix<T> *m, int raw_pos);
      const_col_major_iterator(const Matrix<T> *m, int raw_pos, int *coords);
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
      const T *operator->() const;
      int getRawPos() const;
      int getIdx() const { return idx; }
    };

    /********************************************************/
    /**
     * The sliding is a kind of iterator which traverses the matrix producing
     * sub-matrices following a sliding window similar to dataset.matrix. This
     * iterator couldn't account for CIRCULAR and OUTSIDE values.
     */
    class sliding_window : public Referenced {
      /// A reference to the matrix
      april_utils::SharedPtr< Matrix<T> > m;
      /// Offset coordinates
      int *offset;
      /// subPattern size.
      int *sub_matrix_size;
      /// Step of each dimension for the sliding window.
      int *step;
      /// Number of movements on each dimension (number of steps).
      int *num_steps;
      /// Order of movement for each dimension.
      int *order_step;
      /// Coordinate position of the first component
      int *coords;
      /// Current position at the matrix of the first component
      int raw_pos;
      ///
      int total_size, last_raw_pos;
      bool finished;
      /// auxiliary computation
      int *offset_plus_num_step_by_step;
      /// Number of windows generated by the iterator
      int num_windows;
    public:
      sliding_window();
      sliding_window(Matrix<T> *m,
                     const int *sub_matrix_size=0,
                     const int *offset=0,
                     const int *step=0,
                     const int *num_steps=0,
                     const int *order_step=0);
      sliding_window(const sliding_window &other);
      ~sliding_window();
      sliding_window &operator=(const sliding_window &other);
      sliding_window *next();
      /// This method returns the matrix at the current window position. If a
      /// matrix is given, it must be created before using previous execution of
      /// getMatrix method. WARNING, the matrix is not check to be correct, so be
      /// careful.
      Matrix<T> *getMatrix(Matrix<T> *dest=0);
      bool isEnd() const { return finished; }
      int numWindows() const;
      void setAtWindow(int windex);
      const int *getCoords() const;
      int getNumDim() const;
    };
  
    /********************************************************/
    /**
     * The span iterator traverses the matrix allowing to do a linear traversal of
     * a given dimension. The class allows to generate offset, stride and size
     * values which given the Matrix base pointer could be used to access to any
     * position in a given dimension span.
     */
    class span_iterator {
      friend class Matrix;
      const Matrix<T> *m; ///< A weak reference.
      int raw_pos;
      int *coords, *order;
      int num_iterations;
      struct inverse_sort_compare {
        const Matrix<T> *m;
        inverse_sort_compare(const Matrix<T> *m) : m(m) { }
        bool operator()(const int &a, const int &b) {
          const int a_sz = m->matrixSize[a];
          const int b_sz = m->matrixSize[b];
          if (a_sz == b_sz) {
            if (m->major_order == CblasRowMajor)
              return b < a;
            else
              return a < b;
          }
          // Don't use a trade-off between size and stride, it will be unsafe with
          // transposed matrices
          else
            return a_sz > b_sz;
        }
      };
      //
      void initialize(const Matrix<T> *m, int raw_pos, int dim);
      //
      span_iterator(const Matrix<T> *m, int raw_pos, int dim);
    public:
      span_iterator(const Matrix<T> *m, const int *order);
      span_iterator(const Matrix<T> *m, int dim = -1);
      span_iterator();
      span_iterator(const span_iterator &other);
      ~span_iterator();
      int getOffset() const;
      int getStride() const;
      int getSize() const;
      span_iterator &operator=(const span_iterator &other);
      bool operator==(const span_iterator &other) const;
      bool operator!=(const span_iterator &other) const;
      span_iterator &operator++();
      int numberOfIterations() const;
      void setAtIteration(int idx);
      const int *getDimOrder() const { return order; }
    };

    /********************************************************/
    /**
     * The random access iterator allows to retain the memory pointer, forcing an
     * update between host and device (GPU) memory, and allows to access any
     * position of the matrix given its coordinates.
     */
    class random_access_iterator {
      Matrix<T> *m; ///< A weak reference.
      T *memory;
      int *coords;
    public:
      random_access_iterator(Matrix<T> *m) :
        m(m),
        memory(m->getRawDataAccess()->getPPALForReadAndWrite()),
        coords(new int[m->numDim]) { }
      ~random_access_iterator() { delete[] coords; }
      /// Raw access operation, CAUTION it is dangerous
      T& operator[] (int raw_pos) {
        return memory[raw_pos];
      }
      /// Acces for uni-dimensional matrices
      T& operator() (int i) {
        april_assert(m->numDim == 1);
        int raw_pos = m->computeRawPos(&i);
        return memory[raw_pos];
      }
      /// Acces for bi-dimensional matrices
      T& operator() (int row, int col) {
        april_assert(m->numDim == 2);
        coords[0] = row; coords[1] = col;
        int raw_pos = m->computeRawPos(coords);
        return memory[raw_pos];
      }
      /// Acces for N-dimensional matrices
      T& operator() (int coord0, int coord1, int coord2, ...) {
        april_assert(m->numDim >= 3);
        coords[0] = coord0;
        coords[1] = coord1;
        coords[2] = coord2;
        va_list ap;
        va_start(ap, coord2);
        for(int i=3; i<m->numDim; i++) {
          int coordn = va_arg(ap, int);
          coords[i] = coordn;
        }
        va_end(ap);
        int raw_pos = m->computeRawPos(coords);
        return memory[raw_pos];
      }
      /// Acces for N-dimensional by using a vector of coordinates
      T& operator() (int *coords, int sz) {
        UNUSED_VARIABLE(sz);
        april_assert(m->numDim == sz);
        int raw_pos = m->computeRawPos(coords);
        return memory[raw_pos];
      }
    };
    /**
     * The const random access iterator allows to retain the memory pointer,
     * forcing an update between host and device (GPU) memory, and allows to
     * access any position of the matrix given its coordinates. It is a read-only
     * iterator.
     */
    class const_random_access_iterator {
      const Matrix<T> *m; ///< A weak reference.
      const T *memory;
      int *coords;
    public:
      const_random_access_iterator(const Matrix<T> *m) :
        m(m),
        memory(m->getRawDataAccess()->getPPALForRead()),
        coords(new int[m->numDim]) { }
      ~const_random_access_iterator() { delete[] coords; }
      /// Raw access operation, CAUTION it is dangerous
      const T& operator[] (int raw_pos) const {
        return memory[raw_pos];
      }
      const T& operator() (int i) const {
        april_assert(m->numDim == 1);
        int raw_pos = m->computeRawPos(&i);
        return memory[raw_pos];
      }
      const T& operator() (int row, int col) const {
        april_assert(m->numDim == 2);
        coords[0] = row; coords[1] = col;
        int raw_pos = m->computeRawPos(coords);
        return memory[raw_pos];
      }
      const T& operator() (int coord0, int coord1, int coord2, ...) const {
        april_assert(m->numDim >= 3);
        coords[0] = coord0;
        coords[1] = coord1;
        coords[2] = coord2;
        va_list ap;
        va_start(ap, coord2);
        for(int i=3; i<m->numDim; i++) {
          int coordn = va_arg(ap, int);
          coords[i] = coordn;
        }
        va_end(ap);
        int raw_pos = m->computeRawPos(coords);
        return memory[raw_pos];
      }
      const T& operator() (int *coords, int sz) const {
        UNUSED_VARIABLE(sz);
        april_assert(m->numDim == sz);
        int raw_pos = m->computeRawPos(coords);
        return memory[raw_pos];
      }
    };
  
  private:
    // const version of iterators, for fast end() iterator calls. They are
    // allocated on-demand, so if end() methods are never executed, they
    // don't waste memory space
    mutable iterator end_iterator;
    mutable const_iterator end_const_iterator;
    mutable span_iterator end_span_iterator_;
  
    // NULL constructor
    Matrix() : is_contiguous(NONE),
               end_iterator(), end_const_iterator(),
               end_span_iterator_() { }
    //
    Matrix(int numDim, const int *stride, const int offset,
           const int *matrixSize, const int total_size, const int last_raw_pos,
           april_math::GPUMirroredMemoryBlock<T> *data, const CBLAS_ORDER major_order,
           const bool use_cuda, const bool transposed,
           april_utils::MMappedDataReader *mmapped_data = 0);

    /// Modifies the offset of the matrix. WARNING, this method doesn't check the
    /// new data position, so be sure that it fits in the data pointer size
    void changeSubMatrixData(const int new_offset, const int new_last_raw_pos) {
      offset	 = new_offset;
      last_raw_pos = new_last_raw_pos;
      end_iterator.m	     = 0;
      end_const_iterator.m     = 0;
      end_span_iterator_.m     = 0;
    }

  public:
    /********** Constructors ***********/
    /// Full constructor given numDim, dim, and major_order
    Matrix(int numDim, const int* dim,
           CBLAS_ORDER major_order = CblasRowMajor,
           april_math::GPUMirroredMemoryBlock<T> *data = 0,
           int offset = 0,
           bool transposed = false);
  
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
    /// Sub-matrix constructor of a const matrix. WARNING, this matrices don't
    /// allow writes if clone=false
    Matrix(const Matrix<T> *other,
           const int* coords, const int *sizes,
           bool clone=true);
    /// Destructor
    virtual ~Matrix();
  
    /// Constructor from a MMAP file
    static Matrix<T> *fromMMappedDataReader(april_utils::MMappedDataReader
                                            *mmapped_data);
    /// Writes to a file
    void toMMappedDataWriter(april_utils::MMappedDataWriter *mmapped_data) const;
  
    /// For DEBUG purposes
    void print() const;
  
    /// Modify sizes of matrix
    Matrix<T> *rewrap(const int *new_dims, int len,
                      bool clone_if_not_contiguous=false);

    /// Removes all singleton dimensions
    Matrix<T> *squeeze();
  
    /* Getters and setters */
    int getNumDim() const { return numDim; }
    const int *getDimPtr() const { return matrixSize; }
    const int *getStridePtr() const { return stride; }
    int getDimSize(int i) const { return matrixSize[i]; }
    int getStrideSize(int i) const { return stride[i]; }
    int size() const { return total_size; }
    CBLAS_ORDER getMajorOrder() const { return major_order; }
    bool getTransposedFlag() const { return transposed; }
    bool getIsDataRowOrdered() const {
      return ( (getMajorOrder()==CblasRowMajor && !getTransposedFlag()) ||
               (getMajorOrder()==CblasColMajor &&  getTransposedFlag()) );
    }
    void setUseCuda(bool v) {
      use_cuda = v;
#ifdef USE_CUDA
      if (use_cuda) data->updateMemGPU();
#endif
    }
    bool getCudaFlag() const { return use_cuda; }
    bool isSimple() const {
      return (getIsContiguous())&&(getIsDataRowOrdered());
    }
    /// Indicates if it is a contiguous matrix
    bool getIsContiguous() const;
    /// Returns the offset of first data value (sub-matrix)
    int getOffset() const { return offset; }
    /**********************/
    iterator begin() { return iterator(this); }
    iterator iteratorAt(int c0) {
      april_assert(numDim==1);
      return iterator(this, computeRawPos(&c0), &c0);
    }
    iterator iteratorAt(int c0, int c1) {
      april_assert(numDim==2);
      int aux[2]={c0,c1};
      return iterator(this, computeRawPos(aux), aux);
    }
    iterator iteratorAt(int *coords, int len) {
      UNUSED_VARIABLE(len);
      april_assert(numDim==len);
      return iterator(this, computeRawPos(coords), coords);
    }
    const iterator &end() {
      if (end_iterator.m == 0)
        end_iterator = iterator(this, last_raw_pos+1);
      return end_iterator;
    }
    const span_iterator &end_span_iterator() const {
      if (end_span_iterator_.m == 0)
        end_span_iterator_ = span_iterator(this, last_raw_pos+1, -1);
      return end_span_iterator_;
    }
    /************************/
    const_iterator begin() const { return const_iterator(this); }
    const_iterator iteratorAt(int c0) const {
      april_assert(numDim==1);
      return const_iterator(this, computeRawPos(&c0), &c0);
    }
    const_iterator iteratorAt(int c0, int c1) const {
      april_assert(numDim==2);
      int aux[2]={c0,c1};
      return const_iterator(this, computeRawPos(aux), aux);
    }
    const_iterator iteratorAt(int *coords, int len) const {
      UNUSED_VARIABLE(len);
      april_assert(numDim==len);
      return const_iterator(this, computeRawPos(coords), coords);
    }
    const const_iterator &end() const {
      if (end_const_iterator.m == 0)
        end_const_iterator = const_iterator(this,
                                            last_raw_pos+1); 
      return end_const_iterator;
    }

    /// Symbolic transposition, changes the flag and preserves the major order
    /// flag
    Matrix<T>* transpose();
    /// Changing major order is a different way to perform a transposition, but
    /// taking into account a change in the major_order flag
    Matrix<T>* inMajorOrder(CBLAS_ORDER new_major_order);
    /// Copy only sizes, but not data
    Matrix<T>* cloneOnlyDims() const;
    /// Deep copy
    Matrix<T>* clone() const;
    /// Deep copy with different major_order
    Matrix<T> *clone(CBLAS_ORDER major_order) const;
    /// Shallow copy
    Matrix<T>* shallow_copy();
  
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
    april_math::GPUMirroredMemoryBlock<T> *getRawDataAccess() { return data.get(); }
    const april_math::GPUMirroredMemoryBlock<T> *getRawDataAccess() const { return data.get(); }
  
    bool getCol(int col, T* vec, int vecsize);
    bool putCol(int col, T *vec, int vecsize);
    bool putSubCol(int col, int first_row, T *vec, int vecsize);

    /// Returns true if they have the same dimension
    template<typename O>
    bool sameDim(const Matrix<O> *other) const;
    bool sameDim(const int *dims, const int len) const;

    /// Returns a matrix of one less dimension, with the elements selected for the
    /// given dimension at the given index.  If a matrix is given, it must be
    /// created before using previous execution of select method over the same
    /// dimension. WARNING, the matrix is not check to be correct, so be careful.
    Matrix<T> *select(int dim, int index, Matrix<T> *dest=0);
  
    // Expands current matrix to a diagonal matrix
    Matrix<T> *diagonalize() const;
  
    /**** LAPACK OPERATIONS ****/
    
    // UPDATE GPU OR PPAL IF NEEDED
    void update() {
#ifdef USE_CUDA
      data->forceUpdate(use_cuda);
#endif
    }
  
    /*Matrix<T> **unrolled_kernel=0,
      Matrix<T> **unrolled_this=0);*/
    Matrix<T> *padding(int *begin_padding, int *end_padding, T default_value=T()) const;
    Matrix<T> *padding(int pad_value, T default_value=T()) const;
    
    // SERIALIZATION
    
    /**
     * @brief Writes the Matrix into a stream.
     *
     * The @c options dictionary can contain the following keys:
     *
     * - MatrixIO::TAB_OPTION contains a bool value indicating if writing the
     *   Matrix in a tabulated way (true) or in the APRIL-ANN Matrix format
     *   (false). By default it is false.
     *
     * - MatrixIO::ASCII_OPTION if @c TAB_OPTION=false this key contains a bool
     *   value indicating if the data has to be binary or not. It uses
     *   april_utils::binarizer for binarization purposes. By default it is
     *   true.
     */
    virtual void write(AprilIO::StreamInterface *stream,
                       const april_utils::GenericOptions *options);
    
    /**
     * @brief Reads the Matrix from a stream.
     *
     * @return A Matrix pointer or a NULL pointer if it was impossible to be
     * allocated.
     *
     * The @c options dictionary can contain the following keys:
     *
     * - MatrixIO::TAB_OPTION contains a bool value indicating if read the
     *   Matrix in a tabulated way (true) or in the APRIL-ANN Matrix format
     *   (false). By default it is false.
     *
     * - MatrixIO::ORDER_OPTION this key contains a string with "row_major",
     *   "col_major", or it can be not defined at all. It forces the read()
     *   method to allocate a Matrix in the indicate major order. By default it
     *   is not defined and the major order will be taken from the file in case
     *   @c TAB_OPTION=false or in "row_major" in case @c TAB_OPTION=true.
     *
     * - MatrixIO::DELIM_OPTION if @c TAB_OPTION=true this key contains a string
     *   value with a list of delimitiers. By default it is "\n\r\t,; ".
     *
     * - MatrixIO::EMPTY_OPTION if @c TAB_OPTION=true this key contains a boolean
     *   indicating if empty fields are allowed during the read process. If
     *   @c EMPTY_OPTION=true, this condition allows the parser to find empty
     *   values (e.g. in a CSV file an empty field is detected by this
     *   procedure). By default it is false.
     *
     * - MatrixIO::DEFAULT_OPTION if @c TAB_OPTION=true and @c EMPTY_OPTION=true,
     *   this key contains the T value for cases where the read data is
     *   empty. By default it is T().
     *
     * - MatrixIO::NCOLS_OPTION if @c TAB_OPTION=true this key contains an
     *   int32_t value indicating the number of expected columns in the
     *   Matrix. By default it is 0, which is equals to a not defined state.
     *
     * - MatrixIO::NROWS_OPTION if @c TAB_OPTION=true this key contains an
     *   int32_t value indicating the number of expected rows in the Matrix. By
     *   default it is 0, which is equals to a not defined state.
     *
     * @note When @c TAB_OPTION=true, if not given both @c NCOLS_OPTION and @c
     * NROWS_OPTION the parser will need two passes trough the data, first to
     * compute the number of rows and columns, and second to retrieve the data.
     *
     * @note This method throws different kind of errors.
     */
    static Matrix<T> *read(AprilIO::StreamInterface *stream,
                           const april_utils::GenericOptions *options);
    
  private:
    void allocate_memory(int size);
    void release_memory();
    void initialize(const int *dim);

    static april_utils::constString readULine(AprilIO::StreamInterface *stream,
                                              AprilIO::CStringStream *dest,
                                              bool read_empty = false) {
      // Not needed, it is done in extractULineFromStream: dest->clear(); 
      extractULineFromStream(stream, dest, read_empty);
      return dest->getConstString();
    }

    void writeNormal(AprilIO::StreamInterface *stream,
                     const april_utils::GenericOptions *options);
    
    void writeTab(AprilIO::StreamInterface *stream,
                  const april_utils::GenericOptions *options);

    static Matrix<T> *readNormal(AprilIO::StreamInterface *stream,
                                 const april_utils::GenericOptions *options);
    
    static Matrix<T> *readTab(AprilIO::StreamInterface *stream,
                              const april_utils::GenericOptions *options);
    
    static T getTemplateOption(const april_utils::GenericOptions *options,
                               const char *name, T default_value);
  };

} // namespace basics

#include "sparse_matrix.h"

// must be defined here
#include "matrix_operations.h"

#include "matrix.impl.h"
#include "matrix-iterators.impl.h"
#include "matrix-serialization.impl.h"

#endif // MATRIX_H
