/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include "cmath_overloads.h"

namespace AprilMath {
  
  template<> char Limits<char>::min() { return CHAR_MIN; }
  template<> char Limits<char>::max() { return CHAR_MAX; }

  template<> int32_t Limits<int32_t>::min() { return INT_MIN; }
  template<> int32_t Limits<int32_t>::max() { return INT_MAX; }
  
  template<> float Limits<float>::min() { return FLT_MIN; }
  template<> float Limits<float>::max() { return FLT_MAX; }
  template<> float Limits<float>::epsilon() { return FLT_EPSILON; }
  
  template<> double Limits<double>::min() { return DBL_MIN; }
  template<> double Limits<double>::max() { return DBL_MAX; }
  template<> double Limits<double>::epsilon() { return DBL_EPSILON; }
  
  template<> ComplexF Limits<ComplexF>::min() {
    return ComplexF(FLT_MIN,FLT_MIN);
  }
  template<> ComplexF Limits<ComplexF>::max() {
    return ComplexF(FLT_MAX,FLT_MAX);
  }
  template<> ComplexF Limits<ComplexF>::epsilon() {
    return ComplexF(FLT_EPSILON,FLT_EPSILON);
  }

#define SCALAR_STD_CMATH_MAP_TEMPLATE(NAME,CFUNC)                       \
  template<> APRIL_CUDA_EXPORT float NAME(const float &v) { return CFUNC##f(v); } \
  template<> APRIL_CUDA_EXPORT double NAME(const double &v) { return CFUNC(v); }

  template<> APRIL_CUDA_EXPORT float m_abs(const float &v) { return fabsf(v); }
  template<> APRIL_CUDA_EXPORT float m_abs(const double &v) { return fabs(v); }
  template<> APRIL_CUDA_EXPORT float m_abs(const ComplexF &v) { return v.abs(); }

  template<> APRIL_CUDA_EXPORT float m_sqrt(const float &v) { return sqrtf(v); }
  template<> APRIL_CUDA_EXPORT float m_sqrt(const double &v) { return sqrt(v); }
  template<> APRIL_CUDA_EXPORT float m_sqrt(const ComplexF &v) { return v.sqrtc(); }

  // log overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_log, log);
  
  // SCALAR_MAP_TEMPLATE(m_log, T, T);
  // APRIL_CUDA_EXPORT template<> float m_log(const float &v) {
  //   if (m_abs(v) < NEAR_ZERO) return logf_NZ;
  //   else return logf(v);
  // }
  // APRIL_CUDA_EXPORT template<> double m_log(const double &v) {
  //   if (m_abs(v) < NEAR_ZERO) return log_NZ;
  //   else return log(v);
  // }
  
  // log1p overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_log1p, log1p);
  
  // exp overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_exp, exp);
  
  // pow overload
  template<>
  APRIL_CUDA_EXPORT float m_pow(const float &v, const float &p) {
    return powf(v, p);
  }
  template<>
  APRIL_CUDA_EXPORT double m_pow(const double &v, const double &p) {
    return pow(v, p);
  }

  // cos overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_cos, cos);
  
  // acos overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_acos, acos);

  // cosh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_cosh, cosh);

  // acosh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_acosh, acosh);

  // sin overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_sin, sin);

  // asin overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_asin, asin);

  // sinh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_sinh, sinh);

  // asinh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_asinh, asinh);

  // tan overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_tan, tan);

  // atan overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_atan, atan);

  // tanh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_tanh, tanh);

  // atanh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_atanh, atanh);

#undef SCALAR_STD_CMATH_MAP_TEMPLATE

} // namespace AprilMath
