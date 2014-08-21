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

#include "constString.h"
#include "error_print.h"
#include "referenced.h"

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

namespace april_math {

  template<typename T>
  struct Complex {
    T data[2];
    __host__ __device__ static Complex<T> one_one() {
      return Complex(static_cast<T>(1.0), static_cast<T>(1.0));
    }
    __host__ __device__ static Complex<T> zero_zero() {
      return Complex(static_cast<T>(0.0), static_cast<T>(0.0));
    }
    __host__ __device__ static Complex<T> one_zero() {
      return Complex(static_cast<T>(1.0), static_cast<T>(0.0));
    }
    __host__ __device__ static Complex<T> zero_one() {
      return Complex(static_cast<T>(0.0), static_cast<T>(1.0));
    }
    __host__ __device__ Complex() { data[REAL_IDX] = T(); data[IMG_IDX] = T(); }
    __host__ __device__ Complex(T r) {
      data[REAL_IDX] = r;
      data[IMG_IDX]  = static_cast<T>(0.0);
    }
    __host__ __device__ Complex(T r, T i) {
      data[REAL_IDX] = r;
      data[IMG_IDX]  = i;
    }
    __host__ __device__ ~Complex() { }
    __host__ __device__ Complex(const Complex<T> &other) { *this = other; }
    __host__ __device__ Complex<T> &operator=(const Complex<T> &other) {
      this->data[REAL_IDX] = other.data[REAL_IDX];
      this->data[IMG_IDX]  = other.data[IMG_IDX];
      return *this;
    }
    __host__ __device__ bool operator==(const Complex<T> &other) const {
      Complex<T> r(other - *this);
      return (r.abs() < static_cast<T>(0.0001));
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


  class LuaComplexFNumber : public Referenced {
  public:
    
    LuaComplexFNumber(const ComplexF &number) : Referenced(), number(number) { }
    LuaComplexFNumber(const char *str) : Referenced() {
      float num;
      char  sign='+'; // initialized to avoid compilation warning
      april_utils::constString cs(str);
      STATES state = INITIAL;
      TOKENS token;
      while(state != FINAL && state != ERROR) {
        token = getToken(cs,num,sign);
        switch(state) {
        case INITIAL:
          switch(token) {
          case TOKEN_FLOAT: number.real()=num; state=NUMBER; break;
          case TOKEN_I: number.real()=0.0f; number.img()=1.0f; state=FINAL; break;
          case TOKEN_SIGN: number.real()=0.0f; state=SIGN; break;
          default: state=ERROR;
          }
          break;
        case NUMBER:
          switch(token) {
          case TOKEN_FLOAT: number.img()=num; state=NUMBER_NUMBER; break;
          case TOKEN_I: number.img()=number.real(); number.real()=0.0f; state=FINAL; break;
          case TOKEN_SIGN: state=NUMBER_SIGN; break;
          case TOKEN_END: number.img()=0.0f; state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case SIGN:
          switch(token) {
          case TOKEN_I: number.img()=(sign=='+')?1.0f:-1.0f; state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case NUMBER_NUMBER:
          switch(token) {
          case TOKEN_I: state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case NUMBER_SIGN:
          switch(token) {
          case TOKEN_I: number.img()=(sign=='+')?1.0f:-1.0f; state=FINAL; break;
          default: state=ERROR;
          }
          break;
        default: state=ERROR;
        }
      }
      if (state == ERROR || !cs.empty())
        ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
    }
    
    ComplexF &getValue() { return number; }
    const ComplexF &getValue() const { return number; }

  private:
    ComplexF number;
    
    // Automaton which interprets a string like this regexp: N?[+-]N?i
    enum STATES { INITIAL, NUMBER, SIGN, NUMBER_SIGN, NUMBER_NUMBER,
                  FINAL, ERROR };
    enum TOKENS { TOKEN_FLOAT, TOKEN_SIGN, TOKEN_I, TOKEN_UNKOWN, TOKEN_END };
    TOKENS getToken(april_utils::constString &cs, float &num, char &sign) {
      if (cs.empty()) return TOKEN_END;
      char ch;
      if (cs.extract_float(&num)) return TOKEN_FLOAT;
      if (cs.extract_char(&ch)) {
        if (ch == '+' || ch == '-') { sign=ch; return TOKEN_SIGN; }
        else if (ch == 'i') return TOKEN_I;
      }
      return TOKEN_UNKOWN;
    }
    
  };

} // namespace april_math

#ifdef UNDEF_HOST
#undef __host__
#undef UNDEF_HOST
#endif
#ifdef UNDEF_DEVICE
#undef __device__
#undef UNDEF_DEVICE
#endif

namespace april_utils {
  void aprilPrint(const april_math::ComplexF &v);
  void aprilPrint(const april_math::ComplexD &v);
}

#endif // COMPLEX_NUMBER_H
