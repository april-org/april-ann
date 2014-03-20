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

#include "unused_variable.h"

/***** ITERATORS *****/

template <typename T>
SparseMatrix<T>::iterator::iterator(SparseMatrix<T> *m, int idx) :
  m(m), idx(idx) {
  // IncRef(m);
  values      = m->values->getPPALForReadAndWrite();
  indices     = m->indices->getPPALForReadAndWrite();
  first_index = m->first_index->getPPALForReadAndWrite();
}

template <typename T>
SparseMatrix<T>::iterator::iterator() : m(0), idx(0),
					values(0), indices(0),
					first_index(0) { }

template <typename T>
SparseMatrix<T>::iterator::iterator(const iterator &other) :
  m(other.m),
  idx(other.idx) {
  // IncRef(m);
  values      = m->values->getPPALForReadAndWrite();
  indices     = m->indices->getPPALForReadAndWrite();
  first_index = other.first_index;
}

template <typename T>
SparseMatrix<T>::iterator::~iterator() {
  // if (m) DecRef(m);
}

template <typename T>
typename SparseMatrix<T>::iterator &SparseMatrix<T>::iterator::
operator=(const SparseMatrix<T>::iterator &other) {
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
  idx = other.idx;
  values      = m->values->getPPALForReadAndWrite();
  indices     = m->indices->getPPALForReadAndWrite();
  first_index = other.first_index;
  return *this;
}

template <typename T>
bool SparseMatrix<T>::iterator::operator==(const SparseMatrix<T>::iterator &other) const {
  return m==other.m && idx == other.idx;
}

template <typename T>
bool SparseMatrix<T>::iterator::operator!=(const SparseMatrix<T>::iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
typename SparseMatrix<T>::iterator &SparseMatrix<T>::iterator::operator++() {
  ++idx;
  if (idx < m->nonZeroSize())
    while(*(first_index+1) <= idx) ++first_index;
  return *this;
}

template <typename T>
T &SparseMatrix<T>::iterator::operator*() {
  return values[idx];
}

template <typename T>
T *SparseMatrix<T>::iterator::operator->() {
  return &values[idx];
}

template <typename T>
void SparseMatrix<T>::iterator::getCoords(int &x0, int &x1) const {
  switch(m->sparse_format) {
  case CSC_FORMAT:
    x0 = indices[idx];
    x1 = *first_index;
    break;
  case CSR_FORMAT:
    x0 = *first_index;
    x1 = indices[idx];
    break;
  default:
    x0=-1;
    x1=-1;
  }
}

/*******************************************************************/

template <typename T>
SparseMatrix<T>::const_iterator::const_iterator(const SparseMatrix<T> *m,
						int idx) :
  m(m), idx(idx) {
  // IncRef(m);
  values      = m->values->getPPALForRead();
  indices     = m->indices->getPPALForRead();
  first_index = m->first_index->getPPALForRead();
}

template <typename T>
SparseMatrix<T>::const_iterator::const_iterator() : m(0), idx(0),
						    values(0), indices(0),
						    first_index(0) { }

template <typename T>
SparseMatrix<T>::const_iterator::const_iterator(const const_iterator &other) :
  m(other.m),
  idx(other.idx) {
  // IncRef(m);
  values      = m->values->getPPALForReadAndWrite();
  indices     = m->indices->getPPALForReadAndWrite();
  first_index = other.first_index;
}

template <typename T>
SparseMatrix<T>::const_iterator::const_iterator(const iterator &other) :
  m(other.m),
  idx(other.idx) {
  // IncRef(m);
  values      = m->values->getPPALForRead();
  indices     = m->indices->getPPALForRead();
  first_index = other.first_index;
}

template <typename T>
SparseMatrix<T>::const_iterator::~const_iterator() {
  // if (m) DecRef(m);
}

template <typename T>
typename SparseMatrix<T>::const_iterator &SparseMatrix<T>::const_iterator::
operator=(const SparseMatrix<T>::const_iterator &other) {
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
  idx = other.idx;
  values      = m->values->getPPALForRead();
  indices     = m->indices->getPPALForRead();
  first_index = other.first_index;
  return *this;
}

template <typename T>
typename SparseMatrix<T>::const_iterator &SparseMatrix<T>::const_iterator::
operator=(const SparseMatrix<T>::iterator &other) {
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
  idx = other.idx;
  values      = m->values->getPPALForRead();
  indices     = m->indices->getPPALForRead();
  first_index = other.first_index;
  return *this;
}

template <typename T>
bool SparseMatrix<T>::const_iterator::operator==(const SparseMatrix<T>::const_iterator &other) const {
  return m==other.m && idx == other.idx;
}

template <typename T>
bool SparseMatrix<T>::const_iterator::operator==(const SparseMatrix<T>::iterator &other) const {
  return m==other.m && idx == other.idx;
}

template <typename T>
bool SparseMatrix<T>::const_iterator::operator!=(const SparseMatrix<T>::const_iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
bool SparseMatrix<T>::const_iterator::operator!=(const SparseMatrix<T>::iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
typename SparseMatrix<T>::const_iterator &SparseMatrix<T>::const_iterator::operator++() {
  ++idx;
  if (idx < m->nonZeroSize())
    while(*(first_index+1) <= idx) ++first_index;
  return *this;
}

template <typename T>
const T &SparseMatrix<T>::const_iterator::operator*() const {
  return values[idx];
}

template <typename T>
const T *SparseMatrix<T>::const_iterator::operator->() const {
  return &values[idx];
}

template <typename T>
void SparseMatrix<T>::const_iterator::getCoords(int &x0, int &x1) const {
  switch(m->sparse_format) {
  case CSC_FORMAT:
    x0 = indices[idx];
    x1 = *first_index;
    break;
  case CSR_FORMAT:
    x0 = *first_index;
    x1 = indices[idx];
    break;
  default:
    x0=-1;
    x1=-1;
  }
}
