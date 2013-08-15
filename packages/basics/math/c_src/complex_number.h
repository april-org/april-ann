/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#ifndef COMPLEX_NUMBER_H
#define COMPLEX_NUMBER_H

#include <cmath>

#define REAL_IDX 0
#define IMG_IDX  1

template<typename T>
struct Complex {
  T data[2];
  Complex<T> operator*(const Complex<T> &other) const {
    Complex<T> result;
    result.data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] -
			     this->data[IMG_IDX]*other.data[IMG_IDX]);
    result.data[IMG_IDX]  = (this->data[REAL_IDX]*other.data[IMG_IDX] +
			     this->data[IMG_IDX]*other.data[REAL_IDX]);
    return result;
  }
  Complex<T> operator/(const Complex<T> &other) const {
    T c2_d2 = ( other.data[REAL_IDX]* other.data[REAL_IDX] +
		other.data[IMG_IDX] * other.data[IMG_IDX] );
    Complex<T> result;
    result.data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] +
			     this->data[IMG_IDX]*other.data[IMG_IDX]) / c2_d2;
    result.data[IMG_IDX]  = (this->data[IMG_IDX]*other.data[REAL_IDX] -
			     this->data[REAL_IDX]*other.data[IMG_IDX]) / c2_d2;
    return result;
  }
  Complex<T> operator+(const Complex<T> &other) const {
    Complex<T> result;
    result.data[REAL_IDX] = this->data[REAL_IDX]+other.data[REAL_IDX];
    result.data[IMG_IDX]  = this->data[IMG_IDX]+other.data[IMG_IDX];
    return result;
  }
  Complex<T> operator-(const Complex<T> &other) const {
    Complex<T> result;
    result.data[REAL_IDX] = this->data[REAL_IDX]-other.data[REAL_IDX];
    result.data[IMG_IDX]  = this->data[IMG_IDX]-other.data[IMG_IDX];
    return result;
  }
  Complex<T> operator-() const {
    Complex<T> result;
    result.data[REAL_IDX] = -this->data[REAL_IDX];
    result.data[IMG_IDX]  = -this->data[IMG_IDX];
    return result;
  }
  void conjugate() {
    data[IMG_IDX] = -data[IMG_IDX];
  }
  T real() const { return data[REAL_IDX]; }
  T &real() { return data[REAL_IDX]; }
  T img() const { return data[IMG_IDX]; }
  T &img() { return data[IMG_IDX]; }
  T abs() const { return sqrt( data[REAL_IDX]*data[REAL_IDX] +
			       data[IMG_IDX]*data[IMG_IDX] ); }
  T sqrtc() const { return sqrt( (data[REAL_IDX] + abs()) / 2.0 ); }
};

typedef Complex<float> ComplexF;
typedef Complex<double> ComplexD;
#endif // COMPLEX_NUMBER_H
