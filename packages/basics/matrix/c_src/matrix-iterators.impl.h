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

/***** ITERATORS *****/

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m) : m(m), idx(0), raw_pos(0) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  }
  else coords = 0;
  raw_pos = m->getOffset();
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m, int raw_pos) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    coords = new int[m->getNumDim()];
    m->computeCoords(raw_pos, coords);
  }
  else coords = 0;
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m, int raw_pos, int *coords) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    this->coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) this->coords[i] = coords[i];
  }
  else coords = 0;
  // IncRef(m);
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
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator:: ~iterator() {
  delete[] coords;
  // if (m) DecRef(m);
}

template <typename T>
typename Matrix<T>::iterator &Matrix<T>::iterator::operator=(const Matrix<T>::iterator &other) {
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
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
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    assert(coords != 0);
    m->nextCoordVectorRowOrder(coords, raw_pos);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
T &Matrix<T>::iterator::operator*() {
  return data[raw_pos];
}

template <typename T>
int Matrix<T>::iterator::getRawPos() const {
  return raw_pos;
}

/*******************************************************************/

template <typename T>
Matrix<T>::col_major_iterator::col_major_iterator(Matrix *m) :
  m(m), idx(0), raw_pos(0) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  }
  else coords = 0;
  raw_pos = m->getOffset();
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::col_major_iterator::col_major_iterator(Matrix *m, int raw_pos) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    m->computeCoords(raw_pos, coords);
  }
  else coords = 0;
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::col_major_iterator::col_major_iterator(Matrix *m, int raw_pos, int *coords) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) this->coords[i] = coords[i];
  }
  else coords = 0;
  // IncRef(m);
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
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::col_major_iterator::col_major_iterator(const iterator &other) :
  m(other.m),
  idx(other.idx),
  raw_pos(other.raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  }
  else coords = 0;
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::col_major_iterator::~col_major_iterator() {
  delete[] coords;
  // if (m) DecRef(m);
}

template <typename T>
typename Matrix<T>::col_major_iterator &Matrix<T>::col_major_iterator::operator=(const Matrix<T>::col_major_iterator &other) {
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
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
  // if (m) DecRef(m);
  m = other.m;
  idx = other.idx;
  // IncRef(m);
  idx = other.idx;
  raw_pos = other.raw_pos;
  data = m->getData();
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
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
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    assert(coords != 0);
    m->nextCoordVectorColOrder(coords, raw_pos);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
T &Matrix<T>::col_major_iterator::operator*() {
  return data[raw_pos];
}

template <typename T>
int Matrix<T>::col_major_iterator::getRawPos() const {
  return raw_pos;
}

/*******************************************************************/

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m) :
  m(m), idx(0), raw_pos(0) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  }
  else coords = 0;
  raw_pos = m->getOffset();
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m, int raw_pos) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    coords = new int[m->getNumDim()];
    m->computeCoords(raw_pos, coords);
  }
  else coords = 0;
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m, int raw_pos, int *coords) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
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
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasColMajor) {
    assert(coords != 0);
    m->nextCoordVectorRowOrder(coords, raw_pos);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
const T &Matrix<T>::const_iterator::operator*() const {
  return data[raw_pos];
}

template <typename T>
int Matrix<T>::const_iterator::getRawPos() const {
  return raw_pos;
}

/*******************************************************************/

template <typename T>
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix *m) :
  m(m), idx(0), raw_pos(0) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  }
  else coords = 0;
  raw_pos = m->getOffset();
  data = m->getData();
}

template <typename T>
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix *m, int raw_pos) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    m->computeCoords(raw_pos, coords);
  }
  else coords = 0;
  data = m->getData();
}

template <typename T>
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix *m, int raw_pos, int *coords) :
  m(m), idx(0), raw_pos(raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
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
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix<T>::const_col_major_iterator &other) :
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
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix<T>::iterator &other) :
  m(other.m),
  idx(other.idx), 
  raw_pos(other.raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  }
  else coords = 0;
  data = m->getData();
}

template <typename T>
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const Matrix<T>::const_iterator &other) :
  m(other.m),
  idx(other.idx), 
  raw_pos(other.raw_pos) {
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[m->getNumDim()];
    for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  }
  else coords = 0;
  data = m->getData();
}

/*
template <typename T>
Matrix<T>::const_col_major_iterator::const_col_major_iterator(const iterator &other) :
  m(other.m),
  raw_pos(m.raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  data = m->getData();
}
*/

template <typename T>
Matrix<T>::const_col_major_iterator::~const_col_major_iterator() {
  delete[] coords;
}

template <typename T>
typename Matrix<T>::const_col_major_iterator &Matrix<T>::const_col_major_iterator::operator=(const typename Matrix<T>::const_col_major_iterator &other) {
  m = other.m;
  idx = other.idx;
  raw_pos = other.raw_pos;
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
typename Matrix<T>::const_col_major_iterator &Matrix<T>::const_col_major_iterator::operator=(const typename Matrix<T>::iterator &other) {
  m = other.m;
  idx = other.idx;
  raw_pos = other.raw_pos;
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[other.m->getNumDim()];
    if (other.coords != 0)
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    else m->computeCoords(raw_pos, coords);
  }
  return *this;
}

template <typename T>
typename Matrix<T>::const_col_major_iterator &Matrix<T>::const_col_major_iterator::operator=(const typename Matrix<T>::const_iterator &other) {
  m = other.m;
  idx = other.idx;
  raw_pos = other.raw_pos;
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    coords = new int[other.m->getNumDim()];
    if (other.coords != 0)
      for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
    else m->computeCoords(raw_pos, coords);
  }
  return *this;
}

template <typename T>
bool Matrix<T>::const_col_major_iterator::operator==(const Matrix<T>::const_col_major_iterator &other) const {
  return m==other.m && raw_pos == other.raw_pos;
}

template <typename T>
bool Matrix<T>::const_col_major_iterator::operator==(const Matrix<T>::iterator &other) const {
  return m==other.m && raw_pos == other.raw_pos;
}

template <typename T>
bool Matrix<T>::const_col_major_iterator::operator==(const Matrix<T>::const_iterator &other) const {
  return m==other.m && raw_pos == other.raw_pos;
}

template <typename T>
bool Matrix<T>::const_col_major_iterator::operator!=(const Matrix<T>::const_col_major_iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
bool Matrix<T>::const_col_major_iterator::operator!=(const Matrix<T>::iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
bool Matrix<T>::const_col_major_iterator::operator!=(const Matrix<T>::const_iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
typename Matrix<T>::const_col_major_iterator
&Matrix<T>::const_col_major_iterator::operator++() {
  ++idx;
  if (!m->getIsContiguous() || m->getMajorOrder()==CblasRowMajor) {
    assert(coords != 0);
    m->nextCoordVectorColOrder(coords, raw_pos);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
const T &Matrix<T>::const_col_major_iterator::operator*() const {
  return data[raw_pos];
}

template <typename T>
int Matrix<T>::const_col_major_iterator::getRawPos() const {
  return raw_pos;
}
