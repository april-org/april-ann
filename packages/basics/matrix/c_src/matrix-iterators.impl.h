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
#ifndef MATRIX_ITERATORS_IMPL_H
#define MATRIX_ITERATORS_IMPL_H

#include "matrix.h"
#include "unused_variable.h"

namespace basics {

  /***** ITERATORS *****/

  template <typename T>
  Matrix<T>::iterator::iterator(Matrix<T> *m) : m(m), idx(0), raw_pos(0) {
    if (!m->getIsContiguous() || !m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
    }
    else coords = 0;
    raw_pos = m->getOffset();
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::iterator::iterator(Matrix<T> *m, int raw_pos) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || !m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::iterator::iterator(Matrix<T> *m, int raw_pos, int *coords) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || !m->getIsDataRowOrdered()) {
      this->coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) this->coords[i] = coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::iterator::iterator() : m(0), idx(0), raw_pos(0), coords(0) { }

  template <typename T>
  Matrix<T>::iterator::iterator(const iterator &other) :
    m(other.m),
    idx(other.idx),
    raw_pos(other.raw_pos) {
    if (other.coords != 0) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::iterator:: ~iterator() {
    delete[] coords;
  }

  template <typename T>
  typename Matrix<T>::iterator &Matrix<T>::iterator::
  operator=(const Matrix<T>::iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (other.coords != 0) {
      if (coords==0 || m->numDim != other.m->numDim) {
        delete[] coords;
        coords = new int[other.m->getNumDim()];
      }
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else {
      delete[] coords;
      coords = 0;
    }
    return *this;
  }

  template <typename T>
  bool Matrix<T>::iterator::operator==(const Matrix<T>::iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::iterator::operator!=(const Matrix<T>::iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  typename Matrix<T>::iterator &Matrix<T>::iterator::operator++() {
    ++idx;
    if (coords != 0) m->nextCoordVectorRowOrder(coords, raw_pos);
    else ++raw_pos;
    return *this;
  }

  template <typename T>
  T &Matrix<T>::iterator::operator*() {
    return data[raw_pos];
  }

  template <typename T>
  T *Matrix<T>::iterator::operator->() {
    return &data[raw_pos];
  }

  template <typename T>
  int Matrix<T>::iterator::getRawPos() const {
    return raw_pos;
  }

  /*******************************************************************/

  template <typename T>
  Matrix<T>::col_major_iterator::col_major_iterator(Matrix<T> *m) :
    m(m), idx(0), raw_pos(0) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
    }
    else coords = 0;
    raw_pos = m->getOffset();
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::col_major_iterator::col_major_iterator(Matrix<T> *m, int raw_pos) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::col_major_iterator::col_major_iterator(Matrix<T> *m, int raw_pos, int *coords) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) this->coords[i] = coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::col_major_iterator::col_major_iterator() :
    m(0), idx(0), raw_pos(0), coords(0) { }

  template <typename T>
  Matrix<T>::col_major_iterator::col_major_iterator(const col_major_iterator &other) :
    m(other.m),
    idx(other.idx),
    raw_pos(other.raw_pos) {
    if (other.coords != 0) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::col_major_iterator::col_major_iterator(const iterator &other) :
    m(other.m),
    idx(other.idx),
    raw_pos(other.raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      if (other.coords != 0)
        for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
      else m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::col_major_iterator::~col_major_iterator() {
    delete[] coords;
  }

  template <typename T>
  typename Matrix<T>::col_major_iterator &Matrix<T>::col_major_iterator::operator=(const Matrix<T>::col_major_iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (other.coords != 0) {
      if (coords==0 || m->numDim != other.m->numDim) {
        delete[] coords;
        coords = new int[other.m->getNumDim()];
      }
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else {
      delete[] coords;
      coords = 0;
    }
    return *this;
  }

  template <typename T>
  typename Matrix<T>::col_major_iterator &Matrix<T>::col_major_iterator::operator=(const Matrix<T>::iterator &other) {
    m = other.m;
    idx = other.idx;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[other.m->getNumDim()];
      if (other.coords != 0)
        for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
      else m->computeCoords(raw_pos, coords);
    }
    return *this;
  }

  template <typename T>
  bool Matrix<T>::col_major_iterator::operator==(const Matrix<T>::col_major_iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::col_major_iterator::operator==(const Matrix<T>::iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::col_major_iterator::operator!=(const Matrix<T>::col_major_iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  bool Matrix<T>::col_major_iterator::operator!=(const Matrix<T>::iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  typename Matrix<T>::col_major_iterator &Matrix<T>::col_major_iterator::operator++() {
    ++idx;
    if (coords != 0) m->nextCoordVectorColOrder(coords, raw_pos);
    else ++raw_pos;
    return *this;
  }

  template <typename T>
  T &Matrix<T>::col_major_iterator::operator*() {
    return data[raw_pos];
  }

  template <typename T>
  T *Matrix<T>::col_major_iterator::operator->() {
    return &data[raw_pos];
  }

  template <typename T>
  int Matrix<T>::col_major_iterator::getRawPos() const {
    return raw_pos;
  }

  /*******************************************************************/

  template <typename T>
  Matrix<T>::const_iterator::const_iterator(const Matrix<T> *m) :
    m(m), idx(0), raw_pos(0) {
    if (!m->getIsContiguous() || !m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
    }
    else coords = 0;
    raw_pos = m->getOffset();
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_iterator::const_iterator(const Matrix<T> *m, int raw_pos) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || !m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_iterator::const_iterator(const Matrix<T> *m, int raw_pos, int *coords) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || !m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) this->coords[i] = coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_iterator::const_iterator() :
    m(0), idx(0), raw_pos(0), coords(0) { }

  template <typename T>
  Matrix<T>::const_iterator::const_iterator(const Matrix<T>::const_iterator &other) :
    m(other.m),
    idx(other.idx),
    raw_pos(other.raw_pos) {
    if (other.coords != 0) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_iterator::const_iterator(const Matrix<T>::iterator &other) :
    m(other.m),
    idx(other.idx),
    raw_pos(other.raw_pos) {
    if (other.coords != 0) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_iterator::~const_iterator() {
    delete[] coords;
  }

  template <typename T>
  typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator=(const typename Matrix<T>::const_iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (other.coords != 0) {
      if (coords==0 || m->numDim != other.m->numDim) {
        delete[] coords;
        coords = new int[other.m->getNumDim()];
      }
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else {
      delete[] coords;
      coords = 0;
    }
    return *this;
  }

  template <typename T>
  typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator=(const typename Matrix<T>::iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (other.coords != 0) {
      if (coords==0 || m->numDim != other.m->numDim) {
        delete[] coords;
        coords = new int[other.m->getNumDim()];
      }
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else {
      delete[] coords;
      coords = 0;
    }
    return *this;
  }

  template <typename T>
  bool Matrix<T>::const_iterator::operator==(const Matrix<T>::const_iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::const_iterator::operator==(const Matrix<T>::iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::const_iterator::operator!=(const Matrix<T>::const_iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  bool Matrix<T>::const_iterator::operator!=(const Matrix<T>::iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator++() {
    ++idx;
    if (coords != 0) m->nextCoordVectorRowOrder(coords, raw_pos);
    else ++raw_pos;
    return *this;
  }

  template <typename T>
  const T &Matrix<T>::const_iterator::operator*() const {
    return data[raw_pos];
  }

  template <typename T>
  const T *Matrix<T>::const_iterator::operator->() const {
    return &data[raw_pos];
  }

  template <typename T>
  int Matrix<T>::const_iterator::getRawPos() const {
    return raw_pos;
  }

  /*******************************************************************/

  template <typename T>
  Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix<T> *m) :
    m(m), idx(0), raw_pos(0) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
    }
    else coords = 0;
    raw_pos = m->getOffset();
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix<T> *m,
                                                                int raw_pos) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix<T> *m,
                                                                int raw_pos,
                                                                int *coords) :
    m(m), idx(0), raw_pos(raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) this->coords[i] = coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_col_major_iterator::const_col_major_iterator() :
    m(0), idx(0), raw_pos(0), coords(0) { }

  template <typename T>
  Matrix<T>::const_col_major_iterator::
  const_col_major_iterator(const Matrix<T>::const_col_major_iterator &other) :
    m(other.m),
    idx(other.idx), 
    raw_pos(other.raw_pos) {
    if (other.coords != 0) {
      coords = new int[m->getNumDim()];
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_col_major_iterator::
  const_col_major_iterator(const Matrix<T>::iterator &other) :
    m(other.m),
    idx(other.idx), 
    raw_pos(other.raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      if (other.coords != 0)
        for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
      else m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_col_major_iterator::
  const_col_major_iterator(const Matrix<T>::const_iterator &other) :
    m(other.m),
    idx(other.idx), 
    raw_pos(other.raw_pos) {
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[m->getNumDim()];
      if (other.coords != 0)
        for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
      else m->computeCoords(raw_pos, coords);
    }
    else coords = 0;
    data = m->getData();
  }

  template <typename T>
  Matrix<T>::const_col_major_iterator::~const_col_major_iterator() {
    delete[] coords;
  }

  template <typename T>
  typename Matrix<T>::const_col_major_iterator &Matrix<T>::
  const_col_major_iterator::
  operator=(const typename Matrix<T>::const_col_major_iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (other.coords != 0) {
      if (coords==0 || m->numDim != other.m->numDim) {
        delete[] coords;
        coords = new int[other.m->getNumDim()];
      }
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    }
    else {
      delete[] coords;
      coords = 0;
    }
    return *this;
  }

  template <typename T>
  typename Matrix<T>::const_col_major_iterator &Matrix<T>::
  const_col_major_iterator::operator=(const typename Matrix<T>::iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[other.m->getNumDim()];
      if (other.coords != 0)
        for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
      else m->computeCoords(raw_pos, coords);
    }
    return *this;
  }

  template <typename T>
  typename Matrix<T>::const_col_major_iterator &Matrix<T>::
  const_col_major_iterator::
  operator=(const typename Matrix<T>::const_iterator &other) {
    m = other.m;
    idx = other.idx;
    raw_pos = other.raw_pos;
    data = m->getData();
    if (!m->getIsContiguous() || m->getIsDataRowOrdered()) {
      coords = new int[other.m->getNumDim()];
      if (other.coords != 0)
        for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
      else m->computeCoords(raw_pos, coords);
    }
    return *this;
  }

  template <typename T>
  bool Matrix<T>::const_col_major_iterator::
  operator==(const Matrix<T>::const_col_major_iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::const_col_major_iterator::
  operator==(const Matrix<T>::iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::const_col_major_iterator::
  operator==(const Matrix<T>::const_iterator &other) const {
    return m==other.m && raw_pos == other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::const_col_major_iterator::
  operator!=(const Matrix<T>::const_col_major_iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  bool Matrix<T>::const_col_major_iterator::
  operator!=(const Matrix<T>::iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  bool Matrix<T>::const_col_major_iterator::
  operator!=(const Matrix<T>::const_iterator &other) const {
    return !( (*this) == other );
  }

  template <typename T>
  typename Matrix<T>::const_col_major_iterator
  &Matrix<T>::const_col_major_iterator::operator++() {
    ++idx;
    if (coords != 0) m->nextCoordVectorColOrder(coords, raw_pos);
    else ++raw_pos;
    return *this;
  }

  template <typename T>
  const T &Matrix<T>::const_col_major_iterator::operator*() const {
    return data[raw_pos];
  }

  template <typename T>
  const T *Matrix<T>::const_col_major_iterator::operator->() const {
    return &data[raw_pos];
  }

  template <typename T>
  int Matrix<T>::const_col_major_iterator::getRawPos() const {
    return raw_pos;
  }

  /*******************************************************************/

  template <typename T>
  void Matrix<T>::span_iterator::initialize(const Matrix<T> *m,
                                            int raw_pos,
                                            int dim) {
    if (dim < -2 || dim >= m->numDim) {
      ERROR_EXIT2(128, "Dimension out of bounds, given %d, expected [-1,%d]\n",
                  dim, m->numDim-1);
    }
    this->m       = m;
    this->raw_pos = raw_pos;
    coords = new int[m->numDim];
    order  = new int[m->numDim];
    switch(m->numDim) {
    case 1: order[0] = 0; coords[0] = 0; num_iterations = 1; break;
    case 2:
      coords[0] = 0; coords[1] = 0;
      if (dim == -1) {
        if (m->matrixSize[0] > m->matrixSize[1]) {
          order[0] = 0;
          order[1] = 1;
        }
        else if (m->matrixSize[1] > m->matrixSize[0]) {
          order[0] = 1;
          order[1] = 0;
        }
        else {
          // CUATION: this conditions are critical to work with transposed matrices,
          // in order to ensure the iterator to traverse equally two matrices with
          // different transposition.
          if (m->getMajorOrder() == CblasRowMajor) {
            order[0] = 1;
            order[1] = 0;
          }
          else {
            order[0] = 0;
            order[1] = 1;
          }
        }
      }
      else {
        order[0] = dim;
        order[1] = 1-dim;
      }
      num_iterations = m->matrixSize[order[1]];
      break;
    default:
      if (dim == -1) {
        // compute the best dimension
        for (int i=0; i<m->numDim; ++i) {
          coords[i] = 0;
          order[i] = i;
        }
        april_utils::Sort(order, 0, m->numDim-1, inverse_sort_compare(m));
      }
      else {
        // take given dim as the best and sort other dimensions
        for (int i=0; i<dim; ++i) {
          coords[i+1] = 0;
          order[i+1] = i;
        }
        for (int i=dim+1; i<m->numDim; ++i) {
          coords[i] = 0;
          order[i] = i;
        }
        april_utils::Sort(order, 1, m->numDim-1, inverse_sort_compare(m));
        coords[0] = 0;
        order[0]  = dim;
      }
      num_iterations = 1;
      for (int i=1; i<m->numDim; ++i)
        num_iterations *= m->matrixSize[order[i]];
    }
  }

  template <typename T>
  Matrix<T>::span_iterator::
  span_iterator(const Matrix<T> *m, int raw_pos, int dim) {
    initialize(m, raw_pos, dim);
    m->computeCoords(raw_pos, coords);
  }

  template <typename T>
  Matrix<T>::span_iterator::span_iterator(const Matrix<T> *m, int dim) {
    initialize(m, m->offset, dim);
  }

  template <typename T>
  Matrix<T>::span_iterator::
  span_iterator(const span_iterator &other) :
    m(other.m), raw_pos(other.raw_pos), num_iterations(other.num_iterations) {
    coords = new int[m->getNumDim()];
    order  = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) {
      coords[i] = other.coords[i];
      order[i]  = other.order[i];
    }
  }

  template <typename T>
  Matrix<T>::span_iterator::span_iterator(): m(0),coords(0),order(0) { }

  template <typename T>
  Matrix<T>::span_iterator::~span_iterator() {
    delete[] order;
    delete[] coords;
  }

  template <typename T>
  int Matrix<T>::span_iterator::getOffset() const {
    return raw_pos;
  }

  template <typename T>
  int Matrix<T>::span_iterator::getStride() const {
    return m->stride[order[0]];
  }

  template <typename T>
  int Matrix<T>::span_iterator::getSize() const {
    return m->matrixSize[order[0]];
  }

  template <typename T>
  typename Matrix<T>::span_iterator &Matrix<T>::span_iterator::
  operator=(const Matrix<T>::span_iterator &other) {
    if (m==0 || m->numDim != other.m->numDim) {
      delete[] coords;
      delete[] order;
      coords = new int[other.m->getNumDim()];
      order  = new int[other.m->getNumDim()];
    }
    m = other.m;
    raw_pos = other.raw_pos;
    for (int i=0; i<m->getNumDim(); ++i) {
      coords[i] = other.coords[i];
      order[i]  = other.order[i];
    }
    num_iterations = other.num_iterations;
    return *this;
  }

  template <typename T> bool Matrix<T>::span_iterator::
  operator==(const Matrix<T>::span_iterator &other) const {
    return m==other.m && raw_pos==other.raw_pos;
  }

  template <typename T>
  bool Matrix<T>::span_iterator::
  operator!=(const Matrix<T>::span_iterator &other) const {
    return !((*this)==other);
  }

  template <typename T>
  typename  Matrix<T>::span_iterator &Matrix<T>::span_iterator::
  operator++() {
    switch(m->numDim) {
    case 1: raw_pos = m->last_raw_pos+1; break;
    case 2:
      {
        int j = order[1];
        coords[j] = (coords[j]+1) % m->matrixSize[j];
        if (coords[j] > 0) raw_pos += m->stride[j];
        else raw_pos = m->last_raw_pos+1;
        break;
      }
    default:
      {
        int j = 1, pos;
        do {
          pos = order[j++];
          coords[pos] = (coords[pos]+1) % m->matrixSize[pos];
        } while(j<m->numDim && coords[pos] == 0);
        if (j == m->numDim && coords[pos] == 0) raw_pos = m->last_raw_pos+1;
        else raw_pos = m->computeRawPos(coords);
      }
    }
    return *this;
  }

  template <typename T>
  int Matrix<T>::span_iterator::numberOfIterations() const {
    return num_iterations;
  }

  template <typename T>
  void Matrix<T>::span_iterator::setAtIteration(int idx) {
    if (idx < num_iterations) {
      raw_pos = m->offset;
      coords[order[0]] = 0;
      for (int i=1; i < m->getNumDim(); i++) {
        int j = order[i];
        coords[j]  = idx % m->matrixSize[j];
        idx        = idx / m->matrixSize[j];;
        raw_pos   += m->stride[j]*coords[j];
      }
    }
    else raw_pos = m->last_raw_pos+1;
  }

  /////////////////////////////////////////////////////////////////////////////

  template <typename T>
  Matrix<T>::sliding_window::sliding_window() : Referenced(),
                                                m(0), offset(0), sub_matrix_size(0), step(0),
                                                num_steps(0), order_step(0), coords(0), raw_pos(0),
                                                num_windows(0) { }

  template <typename T>
  Matrix<T>::sliding_window::sliding_window(Matrix<T> *m,
                                            const int *sub_matrix_size,
                                            const int *offset,
                                            const int *step,
                                            const int *num_steps,
                                            const int *order_step) :
    Referenced(),
    m(m),
    offset(new int[m->numDim]),
    sub_matrix_size(new int[m->numDim]),
    step(new int[m->numDim]),
    num_steps(new int[m->numDim]),
    order_step(new int[m->numDim]),
    coords(new int[m->numDim]),
    raw_pos(m->offset),
    finished(false),
    offset_plus_num_step_by_step(new int[m->numDim]),
    num_windows(1)
  {
    if (offset != 0) {
      for (int i=0; i<m->numDim; ++i) {
        this->raw_pos += offset[i]*m->stride[i];
        this->coords[i] = this->offset[i] = offset[i];
      }
    }
    else {
      for (int i=0; i<m->numDim; ++i)
        this->coords[i] = this->offset[i] = 0;
    }
    // default values for arrays if necessary
    if (sub_matrix_size == 0) {
      for (int i=0; i<m->numDim; ++i) {
        this->sub_matrix_size[i] = m->matrixSize[i];
      }
      this->sub_matrix_size[0] = 1;
    }
    else {
      for (int i=0; i<m->numDim; ++i) {
        this->sub_matrix_size[i] = sub_matrix_size[i];
      }
    }
    //
    if (step == 0) {
      for (int i=0; i<m->numDim; ++i) {
        this->step[i] = 1;
      }
    }
    else {
      for (int i=0; i<m->numDim; ++i) {
        this->step[i] = step[i];
      }
    }
    //
    if (num_steps == 0) {
      for (int i=0; i<m->numDim; ++i) {
        this->num_steps[i] = (m->matrixSize[i] - this->sub_matrix_size[i])/this->step[i] + 1;
      }
    }
    else {
      for (int i=0; i<m->numDim; ++i) {
        this->num_steps[i] = num_steps[i];
      }
    }
    //
    if (order_step == 0) {
      for (int i=0; i<m->numDim; ++i) {
        this->order_step[i] = (m->numDim - (i + 1));
      }
    }
    else {
      for (int i=0; i<m->numDim; ++i) {
        this->order_step[i] = order_step[i];
      }
    }
    // last_raw_pos is 0 instead of m->offset because at getMatrix method the
    // resulting matrix last_raw_pos is the addition between current window
    // raw_pos (with m->offset) plus the following last_raw_pos computation
    last_raw_pos = 0;
    total_size   = 1;
    for (int i=0; i<m->numDim; ++i) {
      total_size    = this->sub_matrix_size[i]*total_size;
      last_raw_pos += (this->sub_matrix_size[i]-1)*m->stride[i];
    }
    // Final sanity check and initialization of auxiliary data structures
    for (int i=0; i<m->numDim; ++i) {
      if (this->step[i] < 1) {
        ERROR_EXIT2(128, "Unexpected step value of %d at coordinate %d,"
                    " it must be > 0\n",
                    this->step[i], i);
      }
      if (this->num_steps[i] < 1) {
        ERROR_EXIT2(128, "Unexpected num_steps value of %d at coordinate %d,"
                    " it must be > 0\n",
                    this->num_steps[i], i);
      }
      if (this->offset[i] < 0) {
        ERROR_EXIT1(128, "Unexpected offset value at coordinate %d\n", i);
      }
      if (this->sub_matrix_size[i] < 0) {
        ERROR_EXIT1(128, "Unexpected sub_matrix_size value at coordinate %d\n", i);
      }
      int last = ( this->offset[i] +
                   this->step[i]*(this->num_steps[i]-1) +
                   this->sub_matrix_size[i]);
      if (last > m->matrixSize[i]) {
        ERROR_EXIT1(128, "Overflow at sliding window dimension %d!!!\n", i);
      }
      offset_plus_num_step_by_step[i] = this->offset[i] + this->num_steps[i] * this->step[i];
      num_windows *= this->num_steps[i];
    }
  }

  template <typename T>
  Matrix<T>::sliding_window::sliding_window(const sliding_window &other) :
    Referenced(),
    m(other.m),
    offset(new int[m->numDim]),
    sub_matrix_size(new int[m->numDim]),
    step(new int[m->numDim]),
    num_steps(new int[m->numDim]),
    order_step(new int[m->numDim]),
    coords(new int[m->numDim]),
    raw_pos(other.raw_pos),
    total_size(other.total_size),
    last_raw_pos(other.last_raw_pos),
    finished(other.finished),
    offset_plus_num_step_by_step(new int[m->numDim]),
    num_windows(other.num_windows)
  {
    for (int i=0; i<m->numDim; ++i) {
      sub_matrix_size[i]	= other.sub_matrix_size[i];
      step[i]		= other.step[i];
      num_steps[i]	= other.num_steps[i];
      order_step[i]	= other.order_step[i];
      coords[i]		= other.coords[i];
      offset[i]		= other.offset[i];
      offset_plus_num_step_by_step[i] = other.offset_plus_num_step_by_step[i];
    }
  }

  template <typename T>
  Matrix<T>::sliding_window::~sliding_window() {
    delete[] sub_matrix_size;
    delete[] step;
    delete[] num_steps;
    delete[] order_step;
    delete[] coords;
    delete[] offset;
    delete[] offset_plus_num_step_by_step;
  }
 
  template <typename T>
  typename Matrix<T>::sliding_window::sliding_window &Matrix<T>::sliding_window::
  operator=(const sliding_window &other) {
    if (m.empty() || m->numDim != other.m->numDim) {
      delete[] sub_matrix_size;
      delete[] step;
      delete[] num_steps;
      delete[] order_step;
      delete[] coords;
      delete[] offset;
      delete[] offset_plus_num_step_by_step;
      sub_matrix_size  = new int[other.m->numDim];
      step	     = new int[other.m->numDim];
      num_steps	     = new int[other.m->numDim];
      order_step	     = new int[other.m->numDim];
      coords	     = new int[other.m->numDim];
      offset	     = new int[other.m->numDim];
      offset_plus_num_step_by_step = new int[other.m->numDim];
    }
    m		         = other.m;
    offset	         = other.offset;
    raw_pos	         = other.raw_pos;
    total_size           = other.total_size;
    last_raw_pos         = other.last_raw_pos;
    finished             = other.finished;
    num_windows          = other.num_windows;
    for (int i=0; i<m->numDim; ++i) {
      sub_matrix_size[i] = other.sub_matrix_size[i];
      step[i]		 = other.step[i];
      num_steps[i]	 = other.num_steps[i];
      order_step[i]	 = other.order_step[i];
      coords[i]		 = other.coords[i];
      offset[i]		 = other.offset[i];
      offset_plus_num_step_by_step[i] = other.offset_plus_num_step_by_step[i];
    }	
    return *this;
  }

  template <typename T>
  typename Matrix<T>::sliding_window::sliding_window *Matrix<T>::sliding_window::
  next() {
    int j = 0;
    bool overflow;
    do {
      int pos  = order_step[j];
      int prev = coords[pos];
      coords[pos] += step[pos];
      if (coords[pos] >= offset_plus_num_step_by_step[pos]) {
        coords[pos] = offset[pos];
        raw_pos -= m->stride[pos]*(prev + offset[pos]);
        overflow = true;
      }
      else {
        raw_pos += m->stride[pos]*step[pos];
        overflow = false;
      }
    } while(overflow && ++j<m->numDim);
    if (j == m->numDim && overflow) finished = true;
    return this;
  }

  template <typename T>
  Matrix<T> *Matrix<T>::sliding_window::getMatrix(Matrix<T> *dest) {
    if (finished) return dest;
    if (dest == 0) {
      return new Matrix<T>(m->numDim, m->stride,
                           raw_pos, sub_matrix_size,
                           total_size, last_raw_pos + raw_pos,
                           m->data.get(), m->major_order, m->use_cuda,
                           m->transposed);
    }
    else {
      april_assert(dest->getRawDataAccess() == m->getRawDataAccess());
      dest->changeSubMatrixData(raw_pos, last_raw_pos + raw_pos);
      return dest;
    }
    return 0; // this never happens
  }

  template <typename T>
  int Matrix<T>::sliding_window::numWindows() const {
    return num_windows;
  }

  template <typename T>
  void Matrix<T>::sliding_window::setAtWindow(int windex) {
    if (windex < num_windows) {
      raw_pos = m->offset;
      for (int i=0; i < m->getNumDim();i++) {
        int j = order_step[i];
        coords[j]  = offset[j] + (windex % num_steps[j])*step[j];
        windex     = windex / num_steps[j];
        raw_pos   += m->stride[j]*coords[j];
      }
    }
    else finished = true;
  }

  template <typename T>
  const int *Matrix<T>::sliding_window::getCoords() const {
    return coords;
  }

  template <typename T>
  int Matrix<T>::sliding_window::getNumDim() const {
    return m->getNumDim();
  }
} // namespace basics

#endif // MATRIX_ITERATORS_IMPL_H
