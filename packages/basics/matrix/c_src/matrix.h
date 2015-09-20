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
#include "cmath_overloads.h"
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

namespace Basics {
  
  namespace MatrixIO {
    /// Boolean option key for read/write using tabulated format.
    const char * const TAB_OPTION   = "tab";
    /// Boolean option key for read/write using ascii format.
    const char * const ASCII_OPTION = "ascii";
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
    /// T option key indicating a map of strings into T values.
    const char * const MAP_OPTION = "map";
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
    AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<T> > data;
    AprilUtils::SharedPtr< AprilUtils::MMappedDataReader > mmapped_data;
    /// For CUDA purposes
    bool use_cuda;
    /// To know if it is contiguous
    mutable matrix_contiguous_enum_t is_contiguous;
  
    /// Returns the data pointer for read and write
    T *getData() { return data->getPPALForReadAndWrite(); }
    /// Returns the data pointer for read
    const T *getData() const { return data->getPPALForRead(); }
  
    int getLastRawPos() const { return last_raw_pos; }
    
  public:
    class sliding_window;
    friend class sliding_window;
  
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

    /// Computes the position at data array given it coordinates
    int  computeRawPos(const int *coords) const;
    /// Computes the position at data array given it coordinates
    int  computeRawPos(const int *coords, const int *offset) const;
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
    /**
     * @brief This iterator only traverses the Matrix positions, but doesn't
     * have access to the memory.
     *
     * @todo Implement other iterators as wrappers or derived class from
     * pos_iterator, allowing to share a lot of code.
     */
    class pos_iterator {
      friend class Matrix;
      const Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous,
      /// otherwise it is NULL
      int *coords;
    public:
      pos_iterator(const Matrix<T> *m);
      pos_iterator();
      ~pos_iterator();
      inline pos_iterator &operator=(const pos_iterator &other);
      inline bool          operator==(const pos_iterator &other) const;
      inline bool          operator!=(const pos_iterator &other) const;
      inline pos_iterator &operator++();
      int getRawPos() const { return raw_pos; }
      int getIdx() const { return idx; }
      bool isEnd() const { return raw_pos == m->last_raw_pos+1; }
    };
    // forward declaration
    class const_iterator;
    class iterator {
      friend class const_iterator;
      friend class Matrix;
      Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous,
      /// otherwise it is NULL
      int *coords;
      T *data;
      iterator(Matrix<T> *m);
      iterator(Matrix<T> *m, int raw_pos);
      iterator(Matrix<T> *m, int raw_pos, int *coords);
    public:
      iterator();
      iterator(const iterator &other);
      ~iterator();
      inline iterator &operator=(const iterator &other);
      inline bool      operator==(const iterator &other) const;
      inline bool      operator!=(const iterator &other) const;
      inline iterator &operator++();
      inline T &operator*();
      inline T *operator->();
      int getRawPos() const;
      int getIdx() const { return idx; }
    };
    /*******************************************************/
    class const_iterator {
      friend class Matrix;
      const Matrix<T> *m; ///< A weak reference.
      int idx;
      int raw_pos;
      /// The coords array is only used when the matrix is not congiuous,
      /// otherwise it is NULL
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
      inline const_iterator &operator=(const const_iterator &other);
      inline const_iterator &operator=(const iterator &other);
      inline bool            operator==(const const_iterator &other) const;
      inline bool            operator==(const iterator &other) const;
      inline bool            operator!=(const const_iterator &other) const;
      inline bool            operator!=(const iterator &other) const;
      inline const_iterator &operator++();
      inline const T &operator*() const;
      inline const T *operator->() const;
      int getRawPos() const;
      int getIdx() const { return idx; }
    };
    
    /********************************************************/
    /**
     * The sliding is a kind of iterator which traverses the matrix producing
     * sub-matrices following a sliding window similar to dataset.matrix. This
     * iterator couldn't account for CIRCULAR and OUTSIDE values. This objects
     * IncRefs the given reference, so be careful to check not 0 references in
     * the given matrix object.
     */
    class sliding_window : public Referenced {
      /// A reference to the matrix
      AprilUtils::SharedPtr< Matrix<T> > m;
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
      inline sliding_window &operator=(const sliding_window &other);
      inline sliding_window *next();
      /// This method returns the matrix at the current window position. If a
      /// matrix is given, it must be created before using previous execution of
      /// getMatrix method. WARNING, the matrix is not check to be correct, so be
      /// careful.
      inline Matrix<T> *getMatrix(Matrix<T> *dest=0);
      inline bool isEnd() const { return finished; }
      inline int numWindows() const;
      inline void setAtWindow(int windex);
      inline const int *getCoords() const;
      inline int getNumDim() const;
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
        bool operator()(const int &a, const int &b) const {
          const int a_sz = m->matrixSize[a];
          const int b_sz = m->matrixSize[b];
          if (a_sz == b_sz) {
            return b < a;
          }
          // Don't use a trade-off between size and stride, it will be unsafe with
          // transposed matrices
          else {
            return a_sz > b_sz;
          }
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
      inline int getOffset() const;
      inline int getStride() const;
      inline int getSize() const;
      inline span_iterator &operator=(const span_iterator &other);
      inline bool operator==(const span_iterator &other) const;
      inline bool operator!=(const span_iterator &other) const;
      inline span_iterator &operator++();
      inline int numberOfIterations() const;
      inline void setAtIteration(int idx);
      const int *getDimOrder() const { return order; }
      const int *getCoordinates() const { return coords; }
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
           AprilMath::GPUMirroredMemoryBlock<T> *data,
           const bool use_cuda,
           AprilUtils::MMappedDataReader *mmapped_data = 0);

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
    /// Full constructor given numDim, dim, data, offset, stride
    Matrix(int numDim, const int* dim,
           AprilMath::GPUMirroredMemoryBlock<T> *data = 0,
           int offset = 0,
           const int* stride = 0);
  
    /// Constructor with variadic arguments.
    explicit Matrix(int numDim, int d1, ...);
  
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
    explicit Matrix(const Matrix<T> *other,
                    const int* coords, const int *sizes,
                    bool clone=true);
    /// Destructor
    virtual ~Matrix();

    /// Constructor from a MMAP file
    static Matrix<T> *fromMMappedDataReader(AprilUtils::MMappedDataReader
                                            *mmapped_data);
    /// Writes to a file
    void toMMappedDataWriter(AprilUtils::MMappedDataWriter *mmapped_data) const;
  
    /// For DEBUG purposes
    void print() const;
  
    /// Modify sizes of matrix, returns this if rewrap is not necessary
    Matrix<T> *rewrap(const int *new_dims, int len,
                      bool clone_if_not_contiguous=false);

    /// Modify sizes of matrix, returns always a new instance
    Matrix<T> *constRewrap(const int *new_dims, int len,
                           bool clone_if_not_contiguous=false) const;

    /// Removes all singleton dimensions, always returns a new instance
    Matrix<T> *constSqueeze() const;

    /// Removes all singleton dimensions, returns this when fails squeezing
    Matrix<T> *squeeze();
  
    /* Getters and setters */
    int getNumDim() const { return numDim; }
    const int *getDimPtr() const { return matrixSize; }
    const int *getStridePtr() const { return stride; }
    int getDimSize(int i) const { return matrixSize[i]; }
    int getStrideSize(int i) const { return stride[i]; }
    int size() const { return total_size; }
    void setUseCuda(bool v) {
      use_cuda = v;
#ifdef USE_CUDA
      if (use_cuda) data->updateMemGPU();
#endif
    }
    bool getCudaFlag() const { return use_cuda; }
    bool isSimple() const {
      return (getIsContiguous());
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

    /// Symbolic transposition, changes strides order
    Matrix<T>* transpose();
    /// Symbolic transposition, changes strides order
    Matrix<T>* transpose(int dim1, int dim2);
    /// Copy only sizes, but not data
    Matrix<T>* cloneOnlyDims() const;
    /// Deep copy
    Matrix<T>* clone() const;
    /// Shallow copy
    Matrix<T>* shallowCopy();

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
  
    /// Function to obtain RAW access to data pointer. Be careful with it, you
    /// are losing sub-matrix abstraction.
    AprilMath::GPUMirroredMemoryBlock<T> *getRawDataAccess() { return data.get(); }
    const AprilMath::GPUMirroredMemoryBlock<T> *getRawDataAccess() const { return data.get(); }
  
    bool getCol(int col, T* vec, int vecsize);
    bool putCol(int col, T *vec, int vecsize);
    bool putSubCol(int col, int first_row, T *vec, int vecsize);

    /// Returns true if they have the same dimension
    template<typename O>
    bool sameDim(const Matrix<O> *other) const;
    bool sameDim(const int *dims, const int len) const;

    /**
     * @brief Returns a matrix of one less dimension, with the elements selected
     * for the given dimension at the given index.
     *
     * If @c dest matrix is given, it should be created using a previous
     * execution of select method over the same dimension, otherwise, the
     * behavior of select is undefined.
     *
     * @note <b>WARNING</b>, @c dest matrix correctness is not checked, so, be
     * careful.
     *
     * @code
     * // A tipical use case for a efficient select use with a matAxpy operation
     * // Let A a Basics::Matrix<float> of NxM
     * // Let B a Basics::Matrix<float> of Nx1
     * AprilUtils::SharedPtr< Basics::Matrix<float> > col;
     * for (int i=0; i<A->getDimSize(1); ++i) {
     *   // The first iteration, col=0 and select initialize the col matrix,
     *   // the following iterations the same col matrix will be reused.
     *   col = A->select(1, i, col.get());
     *   matAxpy(col.get(), 1.0, B);
     * }
     * @endcode
     */
    Matrix<T> *select(int dim, int index, Matrix<T> *dest=0);
  
    // Expands current matrix to a diagonal matrix
    Matrix<T> *diagonalize() const;
  
    /**** LAPACK OPERATIONS ****/
    
    // SYNC GPU OR PPAL IF NEEDED
    void sync() {
#ifdef USE_CUDA
      data->forceSync(use_cuda);
#endif
    }

    void sync() const {
#ifdef USE_CUDA
      data->forceSync(use_cuda);
#endif
    }
  
    /*Matrix<T> **unrolled_kernel=0,
      Matrix<T> **unrolled_this=0);*/
    Matrix<T> *padding(int *begin_padding, int *end_padding, T default_value=T()) const;
    Matrix<T> *padding(int pad_value, T default_value=T()) const;
    
    // SERIALIZATION

    virtual const char *luaCtorName() const {
      ERROR_EXIT(128, "Serialization not implemented\n");
      return 0;
    }
    virtual int exportParamsToLua(lua_State *L) {
      AprilUtils::LuaTable t(L);
      AprilUtils::LuaTable sizes(L);
      AprilUtils::LuaTable stride(L);
      for (int i=0; i<getNumDim(); ++i) {
        sizes[i+1] = getDimSize(i);
        stride[i+1] = getStrideSize(i);
      }
      t["sizes"]  = sizes;
      t["stride"] = stride;
      t["offset"] = getOffset();
      t["data"]   = getRawDataAccess();
      t.pushTable(L);
      return 1;
    }
    
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
     *   AprilUtils::binarizer for binarization purposes. By default it is
     *   true.
     */
    virtual void write(AprilIO::StreamInterface *stream,
                       const AprilUtils::LuaTable &options);
    
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
     * - MatrixIO::MAP_OPTION if @c TAB_OPTION=true this key contains a
     *   Lua table with a dictionary of strings to values, allowing to convert
     *   ascii content into numerical data.
     *
     * @note When @c TAB_OPTION=true, if not given both @c NCOLS_OPTION and @c
     * NROWS_OPTION the parser will need two passes trough the data, first to
     * compute the number of rows and columns, and second to retrieve the data.
     *
     * @note This method throws different kind of errors.
     */
    static Matrix<T> *read(AprilIO::StreamInterface *stream,
                           const AprilUtils::LuaTable &options);
    
  private:
    void allocate_memory(int size);
    void release_memory();
    void initialize(const int *dim, int offset=0, const int *stride=0);

    static AprilUtils::constString readULine(AprilIO::StreamInterface *stream,
                                              AprilIO::CStringStream *dest,
                                              bool read_empty = false) {
      // Not needed, it is done in extractULineFromStream: dest->clear(); 
      extractULineFromStream(stream, dest, read_empty);
      return dest->getConstString();
    }

    void writeNormal(AprilIO::StreamInterface *stream,
                     const AprilUtils::LuaTable &options);
    
    void writeTab(AprilIO::StreamInterface *stream,
                  const AprilUtils::LuaTable &options);

    static Matrix<T> *readNormal(AprilIO::StreamInterface *stream,
                                 const AprilUtils::LuaTable &options);
    
    static Matrix<T> *readTab(AprilIO::StreamInterface *stream,
                              const AprilUtils::LuaTable &options);
  };

} // namespace Basics

#include "sparse_matrix.h"

// must be defined here
#include "matrix_ext.h"

#include "matrix.impl.h"
#include "matrix-iterators.impl.h"
#include "matrix-serialization.impl.h"

#include "matrixFloat.h"
#include "matrixDouble.h"
#include "matrixComplexF.h"
#include "matrixInt32.h"
#include "matrixChar.h"
#include "matrixBool.h"

#endif // MATRIX_H
