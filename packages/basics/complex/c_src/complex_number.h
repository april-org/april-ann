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

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#define UNDEF_MATH_DEFINES
#endif

#include <cmath>

#ifdef UNDEF_MATH_DEFINES
#undef _USE_MATH_DEFINES
#undef UNDEF_MATH_DEFINES
#endif

#ifndef NAN
#define NAN sqrtf(-1.0f)
#endif

#define REAL_IDX 0
#define IMG_IDX  1

#ifndef __host__
#define __host__
#define UNDEF_HOST
#endif
#ifndef __device__
#define __device__
#define UNDEF_DEVICE
#endif

template<typename T>
struct Complex {
  T data[2];
  __host__ __device__ static Complex<T> one_one() { return Complex(1.0, 1.0); }
  __host__ __device__ static Complex<T> zero_zero() { return Complex(0.0, 0.0); }
  __host__ __device__ static Complex<T> one_zero() { return Complex(1.0, 0.0); }
  __host__ __device__ static Complex<T> zero_one() { return Complex(0.0, 1.0); }
  __host__ __device__ Complex() { data[REAL_IDX] = T(); data[IMG_IDX] = T(); }
  __host__ __device__ Complex(T r) { data[REAL_IDX] = r; data[IMG_IDX] = 0.0; }
  __host__ __device__ Complex(T r, T i) { data[REAL_IDX] = r; data[IMG_IDX] = i; }
  __host__ __device__ ~Complex() { }
  __host__ __device__ Complex(const Complex<T> &other) { *this = other; }
  __host__ __device__ Complex<T> &operator=(const Complex<T> &other) {
    this->data[REAL_IDX] = other.data[REAL_IDX];
    this->data[IMG_IDX]  = other.data[IMG_IDX];
    return *this;
  }
  __host__ __device__ bool operator==(const Complex<T> &other) const {
    Complex<T> r(other - *this);
    return (r.abs() < 0.0001);
  }
  __host__ __device__ bool operator!=(const Complex<T> &other) const {
    return !(*this == other);
  }
  __host__ __device__ Complex<T> operator*(const Complex<T> &other) const {
    Complex<T> result;
    result.data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] -
			     this->data[IMG_IDX]*other.data[IMG_IDX]);
    result.data[IMG_IDX]  = (this->data[REAL_IDX]*other.data[IMG_IDX] +
			     this->data[IMG_IDX]*other.data[REAL_IDX]);
    return result;
  }
  __host__ __device__ Complex<T> operator/(const Complex<T> &other) const {
    T c2_d2 = ( other.data[REAL_IDX]* other.data[REAL_IDX] +
		other.data[IMG_IDX] * other.data[IMG_IDX] );
    Complex<T> result;
    result.data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] +
			     this->data[IMG_IDX]*other.data[IMG_IDX]) / c2_d2;
    result.data[IMG_IDX]  = (this->data[IMG_IDX]*other.data[REAL_IDX] -
			     this->data[REAL_IDX]*other.data[IMG_IDX]) / c2_d2;
    return result;
  }
  __host__ __device__ Complex<T> &operator+=(const Complex<T> &other) {
    this->data[REAL_IDX] += other.data[REAL_IDX];
    this->data[IMG_IDX]  += other.data[IMG_IDX];
    return *this;
  }
  __host__ __device__ Complex<T> &operator*=(const Complex<T> &other) {
    *this = (*this) * other;
    return *this;
  }
  __host__ __device__ Complex<T> &operator/=(const Complex<T> &other) {
    *this = (*this) / other;
    return *this;
  }
  __host__ __device__ Complex<T> operator+(const Complex<T> &other) const {
    Complex<T> result;
    result.data[REAL_IDX] = this->data[REAL_IDX]+other.data[REAL_IDX];
    result.data[IMG_IDX]  = this->data[IMG_IDX]+other.data[IMG_IDX];
    return result;
  }
  __host__ __device__ Complex<T> operator-(const Complex<T> &other) const {
    Complex<T> result;
    result.data[REAL_IDX] = this->data[REAL_IDX]-other.data[REAL_IDX];
    result.data[IMG_IDX]  = this->data[IMG_IDX]-other.data[IMG_IDX];
    return result;
  }
  __host__ __device__ Complex<T> operator-() const {
    Complex<T> result;
    result.data[REAL_IDX] = -this->data[REAL_IDX];
    result.data[IMG_IDX]  = -this->data[IMG_IDX];
    return result;
  }
  __host__ __device__ bool operator<(const Complex<T> &other) const {
    // FIXME: are you sure?
    return abs() < other.abs();
  }
  __host__ __device__ bool operator>(const Complex<T> &other) const {
    // FIXME: are you sure?
    return abs() > other.abs();
  }
  __host__ __device__ Complex<T> expc() const {
    T expa = exp(data[REAL_IDX]);
    return Complex<T>(expa*cos(data[REAL_IDX]),
		      expa*sin(data[IMG_IDX]));
  }
  __host__ __device__ void conj() {
    data[IMG_IDX] = -data[IMG_IDX];
  }
  __host__ __device__ T real() const { return data[REAL_IDX]; }
  __host__ __device__ T &real() { return data[REAL_IDX]; }
  __host__ __device__ T img() const { return data[IMG_IDX]; }
  __host__ __device__ T &img() { return data[IMG_IDX]; }
  __host__ __device__ T abs() const { return sqrt( data[REAL_IDX]*data[REAL_IDX] +
			       data[IMG_IDX]*data[IMG_IDX] ); }
  __host__ __device__ T sqrtc() const { return sqrt( (data[REAL_IDX] + abs()) / 2.0 ); }
  __host__ __device__ T angle() const {
    T phi;
    if (real() > 0)                     phi = atan(img()/real());
    else if (real() <  0 && img() >= 0) phi = atan(img()/real()) + M_PI;
    else if (real() <  0 && img() <  0) phi = atan(img()/real()) - M_PI;
    else if (real() == 0 && img() >  0) phi =  M_PI/2;
    else if (real() == 0 && img() <  0) phi = -M_PI/2;
    else phi = NAN;
    return phi;
  }
  __host__ __device__ void polar(T &r, T &phi) const {
    r = abs();
    phi = angle();
  }
  // POINTER ACCESS
  __host__ __device__ T *ptr() { return data; }
  __host__ __device__ const T *ptr() const { return data; }
};

typedef Complex<float> ComplexF;
typedef Complex<double> ComplexD;

#ifdef UNDEF_HOST
#undef __host__
#undef UNDEF_HOST
#endif
#ifdef UNDEF_DEVICE
#undef __device__
#undef UNDEF_DEVICE
#endif

#endif // COMPLEX_NUMBER_H
