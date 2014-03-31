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

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "error_print.h"
#include "ignore_result.h"
#include "qsort.h"
#include "swap.h"

template<typename T>
const unsigned int SparseMatrix<T>::MATRIX_BINARY_VERSION = 0x00000001;

template<typename T>
int SparseMatrix<T>::searchIndexOf(const int c0, const int c1) const {
  int idx;
  switch(sparse_format) {
  case CSC_FORMAT:
    if (c1 >= matrixSize[1]) idx = nonZeroSize()+1;
    else idx = doSearchCSCSparseIndexOf(indices, first_index, c0, c1, use_cuda);
    break;
  case CSR_FORMAT:
    if (c0 >= matrixSize[0]) idx = nonZeroSize()+1;
    else idx = doSearchCSRSparseIndexOf(indices, first_index, c0, c1, use_cuda);
    break;
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
    idx=-1;
  }
  return idx;
}

template<typename T>
int SparseMatrix<T>::searchIndexOfFirst(const int c0, const int c1) const {
  int idx;
  switch(sparse_format) {
  case CSC_FORMAT:
    if (c1 >= matrixSize[1]) idx = nonZeroSize()+1;
    else idx = doSearchCSCSparseIndexOfFirst(indices, first_index, c0, c1,
					     use_cuda);
    break;
  case CSR_FORMAT:
    if (c0 >= matrixSize[0]) idx = nonZeroSize()+1;
    else idx = doSearchCSRSparseIndexOfFirst(indices, first_index, c0, c1,
					     use_cuda);
    break;
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
    idx=-1;
  }
  return idx;
}

template <typename T>
struct SparseMatrixIndicesCmp {
  bool operator()(const april_utils::pair<int,T> &a,
		  const april_utils::pair<int,T> &b) {
    return a.first < b.first;
  }
};
template <typename T>
void SparseMatrix<T>::checkSortedIndices(bool sort) {
  const int dense_size = getDenseCoordinateSize();
  const int compressed_size = getCompressedCoordinateSize();
  // for index sort algorithm
  if (indices->getSize() != values->getSize())
    ERROR_EXIT(256, "Different block sizes found in indices and values\n");
  if (static_cast<int>(first_index->getSize()) != dense_size+1)
    ERROR_EXIT1(256, "First index block size must be %d\n", dense_size+1);
  const int *indices_ptr = indices->getPPALForRead();
  const int *first_index_ptr = first_index->getPPALForRead();
  if (first_index_ptr[dense_size] != static_cast<int>(values->getSize()))
    ERROR_EXIT2(256, "Incorrect last value of first_index block, "
		"found %d, expected %u\n", first_index_ptr[dense_size],
		values->getSize());
  april_utils::vector< april_utils::pair<int,T> > aux_index_sorted;
  for (int i=0; i<dense_size; ++i) {
    bool need_sort = false;
    if (first_index_ptr[i] < first_index_ptr[i+1] &&
	(indices_ptr[first_index_ptr[i]] < 0 ||
	 indices_ptr[first_index_ptr[i]] >= compressed_size))
      ERROR_EXIT2(256, "Index out-of-bounds [%d] = %d\n",
		  first_index_ptr[i],
		  indices_ptr[first_index_ptr[i]]);
    for (int j=first_index_ptr[i]+1; j<first_index_ptr[i+1]; ++j) {
      if (indices_ptr[j-1] >= indices_ptr[j]) need_sort = true;
      if (indices_ptr[j] < 0 || indices_ptr[j] >= compressed_size)
	ERROR_EXIT2(256, "Index out-of-bounds [%d] = %d\n",
		    j, indices_ptr[j]);
    }
    if (need_sort) {
      if (!sort) ERROR_EXIT(256, "Found incorrect index order\n");
      aux_index_sorted.clear();
      T *values_ptr = values->getPPALForReadAndWrite();
      int *indices_ptr = indices->getPPALForReadAndWrite();
      for (int j=first_index_ptr[i]; j<first_index_ptr[i+1]; ++j)
	aux_index_sorted.push_back(april_utils::make_pair(indices_ptr[j],
							  values_ptr[j]));
      april_assert(aux_index_sorted.size() > 0);
      april_utils::Sort(aux_index_sorted.begin(),
			0, static_cast<int>(aux_index_sorted.size())-1,
			SparseMatrixIndicesCmp<T>());
      int k=0;
      for (int j=first_index_ptr[i]; j<first_index_ptr[i+1]; ++j, ++k) {
	indices_ptr[j] = aux_index_sorted[k].first;
	values_ptr[j] = aux_index_sorted[k].second;
      }
    }
  }
}

template <typename T>
void SparseMatrix<T>::initialize(int d0, int d1) {
  if (d1 == 1 && sparse_format == CSR_FORMAT)
    ERROR_EXIT(128, "Column vector unable with CSR format\n");
  else if (d0 == 1 && sparse_format == CSC_FORMAT)
    ERROR_EXIT(128, "Row vector unable with CSC format\n");
  total_size    = d0*d1;
  matrixSize[0] = d0;
  matrixSize[1] = d1;
}

/// Allocation of memory for data pointer.
template <typename T>
void SparseMatrix<T>::allocate_memory(int size) {
  unsigned int sz = static_cast<unsigned int>(size);
  values  = new GPUMirroredMemoryBlock<T>(sz);
  indices = new Int32GPUMirroredMemoryBlock(sz);
  first_index = new Int32GPUMirroredMemoryBlock(static_cast<unsigned int>(getDenseCoordinateSize())+1);
  IncRef(values);
  IncRef(indices);
  IncRef(first_index);
}

/// Release of the memory allocated for data pointer.
template <typename T>
void SparseMatrix<T>::release_memory() {
  DecRef(values);
  DecRef(indices);
  DecRef(first_index);
}

/// Default constructor
template <typename T>
SparseMatrix<T>::SparseMatrix(const int d0, const int d1,
			      GPUMirroredMemoryBlock<T> *values,
			      Int32GPUMirroredMemoryBlock *indices,
			      Int32GPUMirroredMemoryBlock *first_index,
			      const SPARSE_FORMAT sparse_format,
			      bool sort) :
  Referenced(), shared_count(0), 
  values(values), indices(indices), first_index(first_index),
  mmapped_data(0),
  sparse_format(sparse_format), use_cuda(false),
  end_iterator(), end_const_iterator() {
  initialize(d0, d1);
  if (this->first_index == 0) {
    if ( (d1 == 1 && sparse_format == CSC_FORMAT) ||
	 (d0 == 1 && sparse_format == CSR_FORMAT) ) {
      this->first_index = new Int32GPUMirroredMemoryBlock(2);
      int *first_index_ptr = this->first_index->getPPALForWrite();
      first_index_ptr[0] = 0;
      first_index_ptr[1] = static_cast<int>(values->getSize());
    }
    else ERROR_EXIT(128, "NULL first_index block only allowed for "
		    "column or row vectors, not matrices\n");
  }
  IncRef(values);
  IncRef(indices);
  IncRef(this->first_index);
  checkSortedIndices(sort);
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const Matrix<T> *other,
			      const SPARSE_FORMAT sparse_format,
                              const T zero) :
  Referenced(), shared_count(0), mmapped_data(0),
  sparse_format(sparse_format), use_cuda(other->getCudaFlag()),
  end_iterator(), end_const_iterator() {
  if (other->getNumDim() != 2)
    ERROR_EXIT(128, "Only allowed for bi-dimensional matrices\n");
  initialize(other->getDimSize(0), other->getDimSize(1));
  //
  int non_zero_size = 0;
  typename Matrix<T>::const_iterator it(other->begin());
  for (int c1=0; c1<other->getDimSize(1); ++c1) {
    for (int c0=0; c0<other->getDimSize(0); ++c0, ++it) {
      if (it == other->end())
	ERROR_EXIT(128, "Unexpected matrix iterator end\n");
      if (zero < *it || *it < -zero) non_zero_size++;
    }
  }
  allocate_memory(non_zero_size);
  int current = 0;
  float *values_ptr    = values->getPPALForWrite();
  int *indices_ptr     = indices->getPPALForWrite();
  int *first_index_ptr = first_index->getPPALForWrite();
  first_index_ptr[0] = 0;
  switch(sparse_format) {
  case CSC_FORMAT:
    {
      typename Matrix<T>::const_col_major_iterator it(other->begin());
      for (int c1=0; c1<other->getDimSize(1); ++c1) {
	for (int c0=0; c0<other->getDimSize(0); ++c0, ++it) {
	  if (zero < *it || *it < -zero) {
	    values_ptr[current]  = *it;
	    indices_ptr[current] = c0;
	    ++current;
	  }
	}
	first_index_ptr[c1+1] = current;
      }
    }
    break; // case CSC_FORMAT
  case CSR_FORMAT:
    {
      typename Matrix<T>::const_iterator it(other->begin());
      for (int c0=0; c0<other->getDimSize(0); ++c0) {
	for (int c1=0; c1<other->getDimSize(1); ++c1, ++it) {
	  if (zero < *it || *it < -zero) {
	    values_ptr[current]  = *it;
	    indices_ptr[current] = c1;
	    ++current;
	  }
	}
	first_index_ptr[c0+1] = current;
      }
    }
    break; // case CSR_FORMAT
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
  }
}

/// Constructor given other matrix, it does a deep copy (clone).
template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> *other,
			      SPARSE_FORMAT sparse_format) :
  Referenced(),
  shared_count(other->shared_count),
  mmapped_data(0),
  use_cuda(other->use_cuda),
  end_iterator(), end_const_iterator() {
  if (sparse_format == NONE_FORMAT) this->sparse_format = other->sparse_format;
  else this->sparse_format = sparse_format;
  initialize(other->matrixSize[0], other->matrixSize[1]);
  allocate_memory(static_cast<int>(other->values->getSize()));
  if (this->sparse_format == other->sparse_format) {
    values->copyFromBlock(other->values, 0, 0, other->values->getSize());
    indices->copyFromBlock(other->indices, 0, 0, other->indices->getSize());
    first_index->copyFromBlock(other->first_index,
			       0, 0, other->first_index->getSize());
  }
  else {
    // Transformation between compressed sparse formats
    ERROR_EXIT(256, "Not implemented!!!\n");
  }
}

/// Constructor for sub-matrix building
template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> *other,
			      const int *coords, const int *sizes,
			      bool clone) :
  Referenced(),
  shared_count(0), mmapped_data(0),
  sparse_format(other->sparse_format),
  use_cuda(other->use_cuda),
  end_iterator(), end_const_iterator() {
  UNUSED_VARIABLE(clone);
  if (sizes[0] + coords[0] > other->matrixSize[0])
    ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
		sizes[0], coords[0], other->matrixSize[0]);
  if (sizes[1] + coords[1] > other->matrixSize[1])
    ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
		sizes[1], coords[1], other->matrixSize[1]);
  initialize(sizes[0], sizes[1]);
  int non_zero_size = 0;
  int x0=0, x1=0;
  for (const_iterator it(other->begin()); it != other->end(); ++it) {
    it.getCoords(x0,x1);
    if (x0 >= coords[0] && x0 < coords[0]+sizes[0] &&
	x1 >= coords[1] && x1 < coords[1]+sizes[1])
      ++non_zero_size;
  }
  allocate_memory(non_zero_size);
  float *values_ptr    = values->getPPALForWrite();
  int *indices_ptr     = indices->getPPALForWrite();
  int *first_index_ptr = first_index->getPPALForWrite();
  int current=0;
  first_index_ptr[0] = 0;
  switch(sparse_format) {
  case CSC_FORMAT:
    {
      int x0=0, x1=0;
      for (const_iterator it(other->begin()); it != other->end(); ++it) {
	it.getCoords(x0,x1);
	if (coords[0] <= x0 && x0 < coords[0]+sizes[0] &&
	    coords[1] <= x1 && x1 < coords[1]+sizes[1]) {
	  values_ptr[current]  = *it;
	  indices_ptr[current] = x0-coords[0];
	  ++current;
	  first_index_ptr[x1-coords[1]+1] = current;
	}
      }
    }
    break;
  case CSR_FORMAT:
    {
      int x0=0, x1=0;
      for (const_iterator it(other->begin()); it != other->end(); ++it) {
	it.getCoords(x0,x1);
	if (coords[0] <= x0 && x0 < coords[0]+sizes[0] &&
	    coords[1] <= x1 && x1 < coords[1]+sizes[1]) {
	  values_ptr[current]  = *it;
	  indices_ptr[current] = x1-coords[1];
	  ++current;
	  first_index_ptr[x0-coords[0]+1] = current;
	}
      }
    }
    break;
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
  }
}

template <typename T>
SparseMatrix<T> *SparseMatrix<T>::
fromMMappedDataReader(april_utils::MMappedDataReader *mmapped_data) {
  SparseMatrix<T> *obj = new SparseMatrix();
  //
  obj->values  = GPUMirroredMemoryBlock<T>::fromMMappedDataReader(mmapped_data);
  obj->indices = Int32GPUMirroredMemoryBlock::fromMMappedDataReader(mmapped_data);
  obj->first_index = Int32GPUMirroredMemoryBlock::fromMMappedDataReader(mmapped_data);
  IncRef(obj->values);
  IncRef(obj->indices);
  IncRef(obj->first_index);
  //
  unsigned int binary_version = *(mmapped_data->get<unsigned int>());
  if (binary_version != MATRIX_BINARY_VERSION)
    ERROR_EXIT1(128,
		"Incorrect binary matrix version from commit number %d\n",
		mmapped_data->getCommitNumber());
  int *aux_size = mmapped_data->get<int>(2);
  obj->matrixSize[0] = aux_size[0];
  obj->matrixSize[1] = aux_size[1];
  obj->total_size    = *(mmapped_data->get<int>());
  obj->sparse_format = *(mmapped_data->get<SPARSE_FORMAT>());
  // NON MAPPED DATA
  obj->use_cuda      = false;
  obj->shared_count  = 0;
  // THE MMAP POINTER
  obj->mmapped_data  = mmapped_data;
  IncRef(obj->mmapped_data);
  //
  return obj;
}

template <typename T>
void SparseMatrix<T>::toMMappedDataWriter(april_utils::MMappedDataWriter
					  *mmapped_data) const {
  values->toMMappedDataWriter(mmapped_data);
  indices->toMMappedDataWriter(mmapped_data);
  first_index->toMMappedDataWriter(mmapped_data);
  mmapped_data->put(&MATRIX_BINARY_VERSION);
  mmapped_data->put(matrixSize, 2);
  mmapped_data->put(&total_size);
  mmapped_data->put(&sparse_format);
}

template <typename T>
SparseMatrix<T>::~SparseMatrix() {
  release_memory();
  if (mmapped_data != 0) DecRef(mmapped_data);
}

template<typename T>
SparseMatrix<T> *SparseMatrix<T>::transpose() {
  SparseMatrix<T> *result =
    new SparseMatrix<T>(matrixSize[1],
                        matrixSize[0],
                        values,
                        indices,
                        first_index,
                        (sparse_format == CSR_FORMAT) ? CSC_FORMAT : CSR_FORMAT,
                        false);
  return result;
}

template<typename T>
SparseMatrix<T> *SparseMatrix<T>::clone(SPARSE_FORMAT sparse_format) const {
  return new SparseMatrix<T>(this, sparse_format);
}

template <typename T>
T& SparseMatrix<T>::operator[] (int i) {
  return values->get(static_cast<unsigned int>(i));
}

template <typename T>
const T& SparseMatrix<T>::operator[] (int i) const {
  return values->get(static_cast<unsigned int>(i));
}

template <typename T>
const T SparseMatrix<T>::operator() (int row, int col) const {
  int idx = searchIndexOf(row,col);
  if (idx < 0) return T();
  return (*values)[idx];
}

template <typename T>
template <typename O>
bool SparseMatrix<T>::sameDim(const SparseMatrix<O> *other) const {
  return ( sparse_format == other->sparse_format &&
	   sameDim(other->getDimSize(0), other->getDimSize(1)) );
}

template <typename T>
template <typename O>
bool SparseMatrix<T>::sameDim(const Matrix<O> *other) const {
  return ( other->getNumDim() == 2 &&
	   sameDim(other->getDimSize(0), other->getDimSize(1)) );
}

template <typename T>
bool SparseMatrix<T>::sameDim(const int row, const int col) const {
  return (row == matrixSize[0]) && (col == matrixSize[1]);
}

template <typename T>
void SparseMatrix<T>::pruneSubnormalAndCheckNormal() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
Matrix<T> *SparseMatrix<T>::toDense(CBLAS_ORDER order) const {
  Matrix<T> *result = new Matrix<T>(2, matrixSize, order);
  typename Matrix<T>::random_access_iterator result_it(result);
  result->zeros();
  int x0=0,x1=0;
  for (const_iterator it(begin()); it != end(); ++it) {
    it.getCoords(x0,x1);
    result_it(x0,x1) = *it;
  }
  return result;
}

template <typename T>
SparseMatrix<T> *SparseMatrix<T>::asVector() const {
  int sz = matrixSize[0]*matrixSize[1];
  const int *this_indices_ptr     = indices->getPPALForRead();
  const int *this_first_index_ptr = first_index->getPPALForRead();
  //
  bool sort=false;
  GPUMirroredMemoryBlock<T> *result_values  = values->clone();
  Int32GPUMirroredMemoryBlock *result_indices =
    new Int32GPUMirroredMemoryBlock(nonZeroSize());
  Int32GPUMirroredMemoryBlock *result_first_index =
    new Int32GPUMirroredMemoryBlock(2);
  //
  int *result_indices_ptr = result_indices->getPPALForWrite();
  int *result_first_index_ptr = result_first_index->getPPALForWrite();
  result_first_index_ptr[0] = 0;
  result_first_index_ptr[1] = nonZeroSize();
  //
  int d0,d1;
  switch(sparse_format) {
  case CSR_FORMAT:
    d0=1; d1=sz;
    for (int row=0; row<matrixSize[0]; ++row)
      for (int i=this_first_index_ptr[row]; i<this_first_index_ptr[row+1]; ++i)
	result_indices_ptr[i] = this_indices_ptr[i] + row*matrixSize[1];
    break;
  case CSC_FORMAT:
    d0=sz; d1=1;
    sort=true;
    for (int col=0; col<matrixSize[1]; ++col)
      for (int i=this_first_index_ptr[col]; i<this_first_index_ptr[col+1]; ++i)
	result_indices_ptr[i] = this_indices_ptr[i] + col*matrixSize[0];
    break;
  default:
    // never gonna happen
    d0=-1; d1=-1; // this avoids compiler warning
    ERROR_EXIT(128, "Incorrect sparse format\n");
  }
  SparseMatrix<T> *result = new SparseMatrix(d0, d1,
					     result_values,
					     result_indices,
					     result_first_index,
					     sparse_format,
					     sort);
  return result;
}
