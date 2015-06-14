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

#ifndef NAN
#define NAN sqrtf(-1.0f)
#endif

#include "binarizer.h"
#include "constString.h"
#include "lua_table.h"
#include "error_print.h"
#include "referenced.h"

#ifdef UNDEF_MATH_DEFINES
#undef _USE_MATH_DEFINES
#undef UNDEF_MATH_DEFINES
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

namespace AprilMath {
  
  /**
   * @brief A class which declares a complex number.
   *
   * This class is intended to be used as template parameter of Basics::Matrix
   * class, allowing to do algebra operations in complex domain.
   */
  template<typename T>
  struct Complex {
    /// The complex number is an array where [0]=REAL, [1]=IMAGINARY.
    T data[2];
    /// Static function for instantiation of \c 1+1i complex number.
    __host__ __device__ static Complex<T> one_one() {
      return Complex(static_cast<T>(1.0), static_cast<T>(1.0));
    }
    /// Static function for instantiation of \c 0+0i complex number.
    __host__ __device__ static Complex<T> zero_zero() {
      return Complex(static_cast<T>(0.0), static_cast<T>(0.0));
    }
    /// Static function for instantiation of \c 1+0i complex number.
    __host__ __device__ static Complex<T> one_zero() {
      return Complex(static_cast<T>(1.0), static_cast<T>(0.0));
    }
    /// Static function for instantiation of 0+1i complex number.
    __host__ __device__ static Complex<T> zero_one() {
      return Complex(static_cast<T>(0.0), static_cast<T>(1.0));
    }
    /// Default constructor, declares \c 0+0i complex number.
    __host__ __device__ Complex() { data[REAL_IDX] = T(); data[IMG_IDX] = T(); }
    /// Construction for \c r+0i complex number, being \c r the argument.
    __host__ __device__ Complex(T r) {
      data[REAL_IDX] = r;
      data[IMG_IDX]  = static_cast<T>(0.0);
    }
    /// Full construction from real (r) and imaginary (i) arguments.
    __host__ __device__ Complex(T r, T i) {
      data[REAL_IDX] = r;
      data[IMG_IDX]  = i;
    }
    __host__ __device__ ~Complex() { }
    __host__ __device__ Complex(const Complex<T> &other) {
      this->data[0] = other.data[0];
      this->data[1] = other.data[1];
    }
    __host__ __device__ Complex<T> &operator=(const Complex<T> &other) {
      this->data[REAL_IDX] = other.data[REAL_IDX];
      this->data[IMG_IDX]  = other.data[IMG_IDX];
      return *this;
    }
    /// Two complex numbers are equal if their difference module is < 0.0001
    __host__ __device__ bool operator==(const Complex<T> &other) const {
      Complex<T> r(other - *this);
      return (r.abs() < static_cast<T>(0.0001));
    }
    /// Defined using negation of operatior ==
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
      this->data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] -
                              this->data[IMG_IDX]*other.data[IMG_IDX]);
      this->data[IMG_IDX]  = (this->data[REAL_IDX]*other.data[IMG_IDX] +
                              this->data[IMG_IDX]*other.data[REAL_IDX]);
      return *this;
    }
    __host__ __device__ Complex<T> &operator/=(const Complex<T> &other) {
      T c2_d2 = ( other.data[REAL_IDX]* other.data[REAL_IDX] +
                  other.data[IMG_IDX] * other.data[IMG_IDX] );
      this->data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] +
                              this->data[IMG_IDX]*other.data[IMG_IDX]) / c2_d2;
      this->data[IMG_IDX]  = (this->data[IMG_IDX]*other.data[REAL_IDX] -
                              this->data[REAL_IDX]*other.data[IMG_IDX]) / c2_d2;
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
    /// Complex numbers are sorted by its modulus.
    __host__ __device__ bool operator<(const Complex<T> &other) const {
      // FIXME: are you sure?
      return abs() < other.abs();
    }
    /// Complex numbers are sorted by its modulus.
    __host__ __device__ bool operator>(const Complex<T> &other) const {
      // FIXME: are you sure?
      return abs() > other.abs();
    }
    /// Exponential operation.
    __host__ __device__ Complex<T> expc() const {
      T expa = exp(data[REAL_IDX]);
      return Complex<T>(expa*cos(data[REAL_IDX]),
                        expa*sin(data[IMG_IDX]));
    }
    /// Conjugate operation, it is done \b IN-PLACE
    __host__ __device__ void conj() {
      data[IMG_IDX] = -data[IMG_IDX];
    }
    __host__ __device__ T real() const { return data[REAL_IDX]; }
    __host__ __device__ T &real() { return data[REAL_IDX]; }
    __host__ __device__ T img() const { return data[IMG_IDX]; }
    __host__ __device__ T &img() { return data[IMG_IDX]; }
    __host__ __device__ T abs() const { return sqrt( data[REAL_IDX]*data[REAL_IDX] +
                                                     data[IMG_IDX]*data[IMG_IDX] ); }
    /// Returns the square root of the complex number.
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
    /// Retruns by reference the modulus \c r and angle \c phi.
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
  
  /**
   * @brief The class LuaComplexFNumber is intended to parse float Complex
   * numbers ( ComplexF ) from a <tt>const char *</tt> coming from Lua.
   *
   * It uses a simple automaton to perform the parsing of the string following
   * this regexp: \c ([+-]N)?([+-]Ni)?
   *
   * @note The automaton in dot format, excluding ERROR state (non expected
   * tokens achieve always ERROR state):
   * \dot
   * digraph ComplexAutomaton {
   * rankdir=LR;
   * lambda [shape=none, label=""];
   * FINAL [shape=doublecircle];
   * lambda -> INITIAL;
   * INITIAL -> SIGN [label="TOKEN_SIGN"];
   * INITIAL -> NUMBER [label="TOKEN_FLOAT"];
   * INITIAL -> FINAL [label="TOKEN_I"];
   * SIGN -> NUMBER [label="TOKEN_FLOAT"];
   * SIGN -> FINAL [label="TOKEN_I"];
   * NUMBER -> R_SIGN [label="TOKEN_SIGN"];
   * NUMBER -> FINAL [label="TOKEN_I"];
   * NUMBER -> FINAL [label="TOKEN_END"];
   * R_SIGN -> R_NUMBER [label="TOKEN_FLOAT"];
   * R_SIGN -> FINAL [label="TOKEN_I"];
   * R_NUMBER -> FINAL [label="TOKEN_I"];
   * }
   * \enddot
   */
  class LuaComplexFNumber : public Referenced {
  public:
    
    LuaComplexFNumber(const ComplexF &number) : Referenced(), number(number) { }
    
    /**
     * @brief Constructor from <tt>const char *</tt>, implements an automaton
     * (FSM).
     *
     * Constructor from <tt>const char *</tt>, implements an automaton
     * (FSM).
     *
     * @see LuaComplexFNumber class for the automaton dot graph.
     */
    LuaComplexFNumber(const char *str) : Referenced() {
#define SET_SIGN(s, n) (n) = ( (s) == '+' ) ? 1.0f : -1.0f
      float num;
      char  sign='+'; // initialized to avoid compilation warning
      AprilUtils::constString cs(str);
      STATES state = INITIAL;
      TOKENS token;
      while(state != FINAL && state != ERROR) {
        token = getToken(cs,num,sign);
        switch(state) {
        case INITIAL:
          switch(token) {
          case TOKEN_SIGN: SET_SIGN(sign, number.real()); state=SIGN; break;
          case TOKEN_FLOAT: number.real()=num; state=NUMBER; break;
          case TOKEN_I: number.real()=0.0f; number.img()=1.0f; state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case SIGN:
          switch(token) {
          case TOKEN_FLOAT: number.real()=number.real()*num; state=NUMBER; break;
          case TOKEN_I: number.img()=number.real(); number.real()=0.0f; state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case NUMBER:
          switch(token) {
          case TOKEN_SIGN: SET_SIGN(sign, number.img()); state=R_SIGN; break;
          case TOKEN_I: number.img()=number.real(); number.real()=0.0f; state=FINAL; break;
          case TOKEN_END: number.img()=0.0f; state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case R_SIGN:
          switch(token) {
          case TOKEN_FLOAT: number.img()=number.img()*num; state=R_NUMBER; break;
          case TOKEN_I: state=FINAL; break;
          default: state=ERROR;
          }
          break;
        case R_NUMBER:
          switch(token) {
          case TOKEN_I: state=FINAL; break;
          default: state=ERROR;
          }
          break;
        default: state=ERROR;
        }
      }
      if (state == ERROR || !cs.empty())
        ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
#undef SET_SIGN
    }
    
    /// Returns the read complex number by reference.
    ComplexF &getValue() { return number; }
    /// Returns the read complex number by const reference.
    const ComplexF &getValue() const { return number; }

  private:
    /// The parsed number result.
    ComplexF number;
    
    /// Automaton states (syntactic structure).
    enum STATES { INITIAL,  ///< Initial state of the automaton.
                  SIGN,     ///< A sign has been read.
                  NUMBER,   ///< A number has been read.
                  R_SIGN,   ///< A sign after the real part.
                  R_NUMBER, ///< A number after the real part (with/without sign).
                  FINAL,    ///< Parsing finished.
                  ERROR     ///< Unexpected symbol or token error.
    };
    /// Automaton tokens (lexical symbols).
    enum TOKENS { TOKEN_FLOAT,  ///< A real number in scientific format.
                  TOKEN_SIGN,   ///< A sign symbol.
                  TOKEN_I,      ///< The i symbol.
                  TOKEN_UNKOWN, ///< Unrecognized token.
                  TOKEN_END     ///< End of the string.
    };
    
    /**
     * @brief Parses a AprilUtils::constString and returns its token type and
     * value.
     *
     * This function receives a
     * AprilUtils::constString and returns its token identifier. Additionally,
     * if the token is a TOKEN_FLOAT, the argument \c num will be filled with
     * the read number; if the token is a TOKEN_SIGN, the argument \c sign will
     * be filled with a '+' or '-' symbol.
     *
     * @note This function uses AprilUtils::constString::operator[] and
     * AprilUtils::constString::extract_float methods to read the char (sign \c
     * '+'/'-'or \c 'i' symbols), or the float number.
     */
    TOKENS getToken(AprilUtils::constString &cs, float &num, char &sign) {
      if (cs.empty()) return TOKEN_END;
      char ch = cs[0];
      if (ch == '+' || ch == '-') { cs.skip(1); sign=ch; return TOKEN_SIGN; }
      else if (ch == 'i') { cs.skip(1); return TOKEN_I; }
      else if (cs.extract_float(&num)) return TOKEN_FLOAT;
      return TOKEN_UNKOWN;
    }
    
  };

} // namespace AprilMath

#ifdef UNDEF_HOST
#undef __host__
#undef UNDEF_HOST
#endif
#ifdef UNDEF_DEVICE
#undef __device__
#undef UNDEF_DEVICE
#endif

namespace AprilUtils {
  void aprilPrint(const AprilMath::ComplexF &v);
  void aprilPrint(const AprilMath::ComplexD &v);

  template<>
  unsigned int binarizer::binary_size<AprilMath::ComplexF>();

  template<>
  unsigned int binarizer::binary_size<AprilMath::ComplexD>();

  template<>
  void binarizer::
  code<AprilMath::ComplexF>(const AprilMath::ComplexF &value, char *b);

  template<>
  void binarizer::
  code<AprilMath::ComplexD>(const AprilMath::ComplexD &value, char *b);

  template<>
  AprilMath::ComplexF binarizer::decode<AprilMath::ComplexF>(const char *b);
  template<>
  AprilMath::ComplexD binarizer::decode<AprilMath::ComplexD>(const char *b);
  ////////////////////////////////////////////////////////////////////////////
  
  // This section allows to push complex numbers from C++ into Lua by calling
  // LuaTable object methods.
  
  template<> AprilMath::ComplexF LuaTable::
  convertTo<AprilMath::ComplexF>(lua_State *L, int idx);
  
  template<> void LuaTable::
  pushInto<const AprilMath::ComplexF &>(lua_State *L,
                                        const AprilMath::ComplexF &value);

  template<> bool LuaTable::
  checkType<AprilMath::ComplexF>(lua_State *L, int idx);
}

#endif // COMPLEX_NUMBER_H
