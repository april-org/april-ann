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
#include <cstdarg>
#include <new> // surprisingly, placement new doesn't work without this
#include "aligned_memory.h"
#include "april_assert.h"
#include "aux_hash_table.h"
#include "cblas_headers.h"
#include "disallow_class_methods.h"
#include "error_print.h"
#include "gpu_mirrored_memory_block.h"
#include "hash_table.h"
#include "mathcore.h"
#include "matrix.h"
#include "maxmin.h"
#include "mmapped_data.h"
#include "qsort.h"
#include "serializable.h"
#include "swap.h"
#include "unused_variable.h"

namespace Basics {

  // CSC or CSR format explained at MKL "Sparse Matrix Storage Formats":
  // http://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-9FCEB1C4-670D-4738-81D2-F378013412B0.htm

  /// The SparseMatrix class represents bi-dimensional sparse matrices, stored as
  /// CSC or CSR format, using zero-based indexing (as in Lua). The structure of
  /// the matrices is resized dynamically. It is not possible to share internal
  /// memory pointers between different matrices, unless transposition operator
  /// which shared the data pointers.
  template <typename T>
  class SparseMatrix : public AprilIO::Serializable {
    APRIL_DISALLOW_ASSIGN(SparseMatrix);
  
    friend class Matrix<T>;
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
    /// non-zero values
    AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<T> > values;
    /// indices for rows (CSC) or columns (CSR)
    AprilUtils::SharedPtr< AprilMath::Int32GPUMirroredMemoryBlock > indices;
    /// size(values) + 1
    AprilUtils::SharedPtr< AprilMath::Int32GPUMirroredMemoryBlock > first_index;
    /// For mmapped matrices
    AprilUtils::SharedPtr< AprilUtils::MMappedDataReader > mmapped_data;
    /// Format type (CSC or CSR)
    SPARSE_FORMAT sparse_format;
    /// For CUDA purposes
    bool use_cuda;
    
    int searchIndexOf(const int c0, const int c1) const;
    int searchIndexOfFirst(const int c0, const int c1) const;
  
  public:

    //////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Builds a sparse matrix from a Dictionary Of Keys (DOK) format
     * https://en.wikipedia.org/wiki/Sparse_matrix#Dictionary_of_keys_.28DOK.29
     */
    class DOKBuilder : public Referenced {
      typedef AprilUtils::uint_pair Key;
      typedef AprilUtils::hash<Key, T> DictType;
      
    public:
      DOKBuilder() : Referenced(), max_row_index(0), max_col_index(0) { }
      
      ~DOKBuilder() { }
      
      void insert(const Key &coords, const T &value) {
        if (coords.first > max_row_index) max_row_index = coords.first;
        if (coords.second > max_col_index) max_col_index = coords.second;
        data.insert(coords, value);
      }
      
      void insert(unsigned int row, unsigned int col, const T &value) {
        insert(Key(row,col), value);
      }

      /**
       * @brief Builds a sparse matrix by sorting all the coordinates and
       * declaring the CSR/CSC vectors: values, indices and first_index.
       */
      SparseMatrix<T> *build(unsigned int d0=0, unsigned int d1=0,
                             SPARSE_FORMAT format = CSR_FORMAT) const {
        if (d0 == 0) d0 = max_row_index+1;
        if (d1 == 0) d1 = max_col_index+1;
        if (d0 <= max_row_index) {
          ERROR_EXIT2(128, "Improper number of rows expected > %u, given %u\n",
                      max_row_index, d0);
        }
        if (d1 <= max_col_index) {
          ERROR_EXIT2(128, "Improper number of columns expected > %d, given %u\n",
                      max_col_index, d1);
        }
        //
        const Key sizes(d0, d1);
        // put all coordinates of the dictionary into a vector, it will be
        // sorted after to build the result sparse matrix
        AprilUtils::vector<Key> coordinates(data.size());
        unsigned int i=0;
        for (typename DictType::const_iterator it = data.begin();
             it != data.end(); ++it) {
          if (it->second != T(0.0f) && it->second != T(-0.0f)) {
            coordinates[i++] = it->first;
          }
        }
        coordinates.resize(i);
        const unsigned int NNZ = coordinates.size();
        // depending in the format, the privateBuild template receives
        // CSRCompare() or CSCCompare()
        if (format == CSR_FORMAT) {
          return privateBuild(sizes, NNZ, coordinates, CSRCompare());
        }
        else if (format == CSC_FORMAT) {
          return privateBuild(sizes, NNZ, coordinates, CSCCompare());
        }
        else {
          ERROR_EXIT1(128, "Not recognized sparse format: %d\n", format);
        }
        return 0;
      }
      
    private:
      DictType data;
      unsigned int max_row_index, max_col_index;
      
      struct CSRCompare {
        unsigned int getDense(const Key &k) const { return k.first; }
        unsigned int getSparse(const Key &k) const { return k.second; }
        SPARSE_FORMAT getFormat() const { return CSR_FORMAT; }
        bool operator()(const Key &a, const Key &b) const {
          if (a.first < b.first) return true;
          else if (a.first > b.first) return false;
          else return a.second < b.second;
        }
      };
      
      struct CSCCompare {
        unsigned int getDense(const Key &k) const { return k.second; }
        unsigned int getSparse(const Key &k) const { return k.first; }
        SPARSE_FORMAT getFormat() const { return CSC_FORMAT; }
        bool operator()(const Key &a, const Key &b) const {
          if (a.second < b.second) return true;
          else if (a.second > b.second) return false;
          else return a.first < b.first;
        }
      };

      /**
       * @brief This private build template allows to factorize code between
       * both CSR and CSC formats.
       *
       * It receives a CMP type which is an instance of CSRCompare or CSCCompare
       * classes.
       */
      template<typename CMP>
      SparseMatrix *privateBuild(const Key &sizes,
                                 const unsigned int NNZ,
                                 AprilUtils::vector<Key> &coordinates,
                                 CMP compare) const {
        AprilMath::FloatGPUMirroredMemoryBlock *values =
          new AprilMath::FloatGPUMirroredMemoryBlock(NNZ);
        AprilMath::Int32GPUMirroredMemoryBlock *indices =
          new AprilMath::Int32GPUMirroredMemoryBlock(NNZ);
        AprilMath::Int32GPUMirroredMemoryBlock *first_index =
          new AprilMath::Int32GPUMirroredMemoryBlock(compare.getDense(sizes)+1);
        // build the matrix by sorting the coordinates (CSR or CSC order) and
        // filling values, indices and first_index memory blocks
        (*first_index)[0] = 0;
        AprilUtils::Sort(coordinates.begin(), coordinates.size(), compare);
        unsigned int j=0;
        for (unsigned int i=0; i<NNZ; ++i) {
          const Key &k = coordinates[i];
          // this loop is for the case of several empty runs
          while(j < compare.getDense(k)) {
            ++j;
            (*first_index)[j] = i;
          }
          (*values)[i] = *data.find(k);
          (*indices)[i] = static_cast<int32_t>(compare.getSparse(k));
        }
        // process all remaining empty runs
        while(j < compare.getDense(sizes)) {
          ++j;
          (*first_index)[j] = NNZ;
        }
        return new SparseMatrix<T>(static_cast<int>(sizes.first),
                                   static_cast<int>(sizes.second),
                                   values,
                                   indices,
                                   first_index,
                                   compare.getFormat());
      }
      
    }; // class DOKBuilder
    
    //////////////////////////////////////////////////////////////////////////
    
    /// Returns if the matrix is a vector
    bool isVector() const { return ( (matrixSize[0]==1) ||
                                     (matrixSize[1]==1) ); }
    bool isColVector() const { return matrixSize[1]==1; }
  
    /********* Iterators for Matrix traversal *********/
    // forward declaration
    class const_iterator;
    class iterator {
      friend class const_iterator;
      friend class SparseMatrix;
      SparseMatrix<T> *m; // A weak reference.
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
      const SparseMatrix<T> *m; // A weak reference.
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
                 AprilMath::GPUMirroredMemoryBlock<T> *values,
                 AprilMath::Int32GPUMirroredMemoryBlock *indices,
                 AprilMath::Int32GPUMirroredMemoryBlock *first_index,
                 const SPARSE_FORMAT sparse_format = CSR_FORMAT,
                 bool sort=false);
    
    /// Constructor given other matrix, it does a deep copy (clone).
    SparseMatrix(const SparseMatrix<T> *other,
                 SPARSE_FORMAT sparse_format = NONE_FORMAT);
    /// Sub-matrix constructor, makes a deep copy of the given matrix slice
    SparseMatrix(const SparseMatrix<T> *other,
                 const int *coords, const int *sizes, bool clone=true);
    /// Destructor
    virtual ~SparseMatrix();
    
    /// Constructor given a dense Matrix, it returns a SparseMatrix
    static SparseMatrix<T> *fromDenseMatrix(const Matrix<T> *other,
                                            const SPARSE_FORMAT
                                            sparse_format = CSR_FORMAT,
                                            const T zero = T());
    
    /// Constructor from a MMAP file
    static SparseMatrix<T> *fromMMappedDataReader(AprilUtils::MMappedDataReader
                                                  *mmapped_data);
    /// Writes to a file
    void toMMappedDataWriter(AprilUtils::MMappedDataWriter *mmapped_data) const;

    /* Getters and setters */
    int getNumDim() const { return numDim; }
    const int *getDimPtr() const { return matrixSize; }
    int getDimSize(int i) const { return matrixSize[i]; }
    int size() const { return total_size; }
    // FIXME: use an attribute to improve the efficiency of this call
    /// Returns the Number of Non-Zero elements.
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
    // UPDATE GPU OR PPAL IF NEEDED
    void update() {
#ifdef USE_CUDA
      values->forceUpdate(use_cuda);
      indices->forceUpdate(use_cuda);
      first_index->forceUpdate(use_cuda);
#endif
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
    Matrix<T> *toDense() const;
  
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
    /// you are losing sub-matrix abstraction.
    AprilMath::GPUMirroredMemoryBlock<T> *getRawValuesAccess() { return values.get(); }
    AprilMath::Int32GPUMirroredMemoryBlock *getRawIndicesAccess() { return indices.get(); }
    AprilMath::Int32GPUMirroredMemoryBlock *getRawFirstIndexAccess() { return first_index.get(); }

    const AprilMath::GPUMirroredMemoryBlock<T> *getRawValuesAccess() const { return values.get(); }
    const AprilMath::Int32GPUMirroredMemoryBlock *getRawIndicesAccess() const { return indices.get(); }
    const AprilMath::Int32GPUMirroredMemoryBlock *getRawFirstIndexAccess() const { return first_index.get(); }
  
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
  
    static SparseMatrix<T> *diag(int N, T value=T(),
                                 SPARSE_FORMAT sparse_format = CSR_FORMAT) {
      unsigned int uN = static_cast<unsigned int>(N);
      SparseMatrix<T> *result;
      AprilMath::GPUMirroredMemoryBlock<T> *values = new AprilMath::GPUMirroredMemoryBlock<T>(uN);
      AprilMath::Int32GPUMirroredMemoryBlock *indices = new AprilMath::Int32GPUMirroredMemoryBlock(uN);
      AprilMath::Int32GPUMirroredMemoryBlock *first_index = new AprilMath::Int32GPUMirroredMemoryBlock(uN+1);
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
      AprilMath::GPUMirroredMemoryBlock<T> *values = new AprilMath::GPUMirroredMemoryBlock<T>(uN);
      AprilMath::Int32GPUMirroredMemoryBlock *indices = new AprilMath::Int32GPUMirroredMemoryBlock(uN);
      AprilMath::Int32GPUMirroredMemoryBlock *first_index = new AprilMath::Int32GPUMirroredMemoryBlock(uN+1);
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

    static SparseMatrix<T> *diag(AprilMath::GPUMirroredMemoryBlock<T> *values,
                                 SPARSE_FORMAT sparse_format = CSR_FORMAT) {
      unsigned int uN = values->getSize();
      int N = static_cast<int>(uN);
      SparseMatrix<T> *result;
      AprilMath::Int32GPUMirroredMemoryBlock *indices = new AprilMath::Int32GPUMirroredMemoryBlock(uN);
      AprilMath::Int32GPUMirroredMemoryBlock *first_index = new AprilMath::Int32GPUMirroredMemoryBlock(uN+1);
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
    
    /// This method converts the caller SparseMatrix in a vector, unrolling the
    /// dimensions in row-major order. If the SparseMatrix is in CSR format, the
    /// resulting vector is a row vector, otherwise, it is a column vector.
    SparseMatrix<T> *asVector() const;  

    
    // SERIALIZATION

    /**
     * @brief Reads the SparseMatrix from a stream.
     *
     * Any key/value in @c options dictionary will be ignored.
     */
    static SparseMatrix<T> *read(AprilIO::StreamInterface *stream,
                                 const AprilUtils::LuaTable &options);

    /**
     * @brief Writes the SparseMatrix into a stream.
     *
     * The @c options dictionary can contain the following keys:
     *
     * - MatrixIO::ASCII_OPTION key contains a bool value indicating if the data
     *   has to be binary or not. It uses AprilUtils::binarizer for
     *   binarization purposes. By default it is true.
     */
    virtual void write(AprilIO::StreamInterface *stream,
                       const AprilUtils::LuaTable &options);

    
  private:
    
    void allocate_memory(int size);
    void release_memory();
    void initialize(int d0, int d1);

    static AprilUtils::constString readULine(AprilIO::StreamInterface *stream,
                                              AprilIO::CStringStream *dest) {
      // Not needed, it is done in extractULineFromStream: dest->clear(); 
      extractULineFromStream(stream, dest);
      return dest->getConstString();
    }
  };

} // namespace Basics

// must be defined here
#include "matrix_ext.h"

#include "sparse_matrix.impl.h"
#include "sparse_matrix-iterators.impl.h"
#include "sparse_matrix-serialization.impl.h"

#include "sparse_matrixFloat.h"

#endif // SPARSE_MATRIX_H
