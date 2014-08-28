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
#ifndef CMATH_OVERLOADS_H
#define CMATH_OVERLOADS_H
#include <cfloat>
#include <cmath>
#include <climits>
#include "complex_number.h"
#include "cuda_utils.h"
#include "error_print.h"
#include "unused_variable.h"

#define NEAR_ZERO             1e-6f
#define DERIVATIVE_SATURATION 17.0f

/*
  This file contains math operators overloaded to work with basic numeric types
  of APRIL-ANN, and exported to CUDA if it is compiled with definition of
  USE_CUDA constant.
*/
namespace april_math {
  
  typedef float (*m_float_unary_float_map_t)(float);
  typedef double (*m_double_unary_double_map_t)(double);
  typedef ComplexF (*m_complexf_unary_complexf_map_t)(ComplexF);
  typedef float (*m_float_unary_double_map_t)(double);
  typedef float (*m_float_unary_complexf_map_t)(ComplexF);
  
  const float  logf_NZ = logf(NEAR_ZERO);
  const double log_NZ  = log(NEAR_ZERO);
  
  ///////////////// NUMERIC LIMITS /////////////////
  
  template<typename T>
  class Limits {
  public:
    static T min() { return T(); }
    static T max() { return T(); }
    static T epsilon() { return T(); }
  };
  
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

  ///////////////// NAN CHECK /////////////////
  
  template<typename T>
  APRIL_CUDA_EXPORT bool m_isnan(const T &v) {
    return v != v; // by definition, a NAN is always different of any other
                   // value, even another NAN
  }
  
  //////////////// MATH SCALAR MAP FUNCTIONS ////////////////
  
#define SCALAR_MAP_TEMPLATE(NAME, T_IN, T_OUT)  \
  template<typename T>                          \
  APRIL_CUDA_EXPORT T_OUT NAME(const T_IN &v) { \
    UNUSED_VARIABLE(v);                         \
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");       \
    return T_OUT();                             \
  }
#define SCALAR_STD_CMATH_MAP_TEMPLATE(NAME,CFUNC)                       \
  SCALAR_MAP_TEMPLATE(NAME, T, T);                                      \
  template<> APRIL_CUDA_EXPORT float NAME(const float &v) { return CFUNC##f(v); } \
  template<> APRIL_CUDA_EXPORT double NAME(const double &v) { return CFUNC(v); }
  
  // abs overload
  SCALAR_MAP_TEMPLATE(m_abs, T, float);
  template<> APRIL_CUDA_EXPORT float m_abs(const float &v) { return fabsf(v); }
  template<> APRIL_CUDA_EXPORT float m_abs(const double &v) { return fabs(v); }
  template<> APRIL_CUDA_EXPORT float m_abs(const ComplexF &v) { return v.abs(); }
  
  // sqrt overload
  SCALAR_MAP_TEMPLATE(m_sqrt, T, float);
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
  template<typename T>
  APRIL_CUDA_EXPORT T m_pow(const T &v, const T &p) {
    UNUSED_VARIABLE(v);
    UNUSED_VARIABLE(p);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return T();
  }
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

  //
  template<typename T>
  APRIL_CUDA_EXPORT T m_identity(const T &a) {
    return a;
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_plogp(const T &x) {
    return ((x) > T(0.0f) || (x) < T(0.0f)) ? (x) * m_log(x) : (x);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_sign(const T &x) {
    return ((x)<T(0.0f)) ? T(-1.0f) : ( ((x)>T(0.0f)) ? T(1.0f) : T(0.0f) );
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_complement(const T &x) {
    return (T(1.0f) - (x));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_clamp(const T &x,
                              const T &min,
                              const T &max) {
    return ((x)<min?min:((x)>max?max:x));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_sigmoid(const T &numerator,
                                const T &value) {
    (numerator) / (m_exp(-(value))+T(1.0f));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_logistic(const T &value) {
    return m_sigmoid(T(1.0f), value);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_antysim_logistic(const T &value) {
    return m_sigmoid(T(2.0f), value) - T(1.0f);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_loglogistic(const T &value) {
    // The value of -log1p(exp(x)) when X is negative and large, is
    // approximately X
    return ( (value)<T(-10.0f)) ? (value) : (-m_log1p(m_exp(-(value))));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_softsign(const T &value) {
    return value / (T(1.0f) + m_abs(value));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_softplus(const T &value) {
    return m_log1p(m_exp(value));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_relu(const T &value) {
    return (value > T(0.0f)) ? (value) : T(0.0f);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_lt(const T &a, const T &b) {
    if (a < b) return T(1.0f);
    else return T(0.0f);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_gt(const T &a, const T &b) {
    if (b < a) return T(1.0f);
    else return T(0.0f);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_eq(const T &a, const T &b) {
    if (m_isnan(a)) {
      if (m_isnan(b)) return T(1.0f);
      else return T(0.0f);
    }
    else {
      if (a == b) return T(1.0f);
      else return T(0.0f);
    }
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_neq(const T &a, const T &b) {
    if (m_eq(a,b) == T(1.0f)) return T(0.0f);
    else return T(1.0f);
  }
  template<typename T>
  APRIL_CUDA_EXPORT bool m_relative_equals(const T &a,
                                           const T &b,
                                           const T &TH) {
    if (a == T(0.0f) && b == T(0.0f)) return true;
    return 2.0 * ( m_abs(a - b) / (m_abs(a) + m_abs(b)) ) < TH;
  }
  
  // DERIVATIVES
  
  template<typename T>
  APRIL_CUDA_EXPORT T m_logistic_der(const T &input,
                                     const T &output) {
    UNUSED_VARIABLE(input);
    float value = m_clamp(output, NEAR_ZERO, T(1.0f) - NEAR_ZERO);
    return value * (T(1.0f) - value);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_antisym_logistic_der(const T &input,
                                             const T &output) {
    UNUSED_VARIABLE(input);
    T value = m_clamp(output, T(-1.0f) + NEAR_ZERO, T(1.0f) - NEAR_ZERO);
    return T(0.5f) * (T(1.0f) - (value*value));
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_softsign_der(const T &input,
                                     const T &output) {
    UNUSED_VARIABLE(input);
    T value = m_clamp(output, T(-1.0f) + NEAR_ZERO, T(1.0f) - NEAR_ZERO);
    T aux   = T(1.0f) + absolute_value(value);
    return T(1.0f) / (aux * aux);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_softplus_der(const T &input,
                                     const T &output) {
    UNUSED_VARIABLE(output);
    T value = m_logistic(input);
    return value;
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_relu_der(const T &input,
                                 const T &output) {
    UNUSED_VARIABLE(output);
    return (input > T(0.0f)) ? T(1.0f) : T(0.0f);
  }
  template<typename T>
  APRIL_CUDA_EXPORT T m_clamp_der(const T &input,
                                  const T &output,
                                  const T &inf,
                                  const T &sup) {
    UNUSED_VARIABLE(output);
    return (input < inf || input > sup) ? T(0.0f) : T(1.0f);
  }
  
  //////////////// MATH SCALAR REDUCE FUNCTIONS ////////////////
  template<typename T>
  APRIL_CUDA_EXPORT T r_max(const T &a, const T &b) {
    return (a<b) ? (b) : (a);
  }

  template<typename T>
  APRIL_CUDA_EXPORT T r_min(const T &a, const T &b) {
    return (a<b) ? (a) : (b);
  }

  template<typename T>
  APRIL_CUDA_EXPORT T r_max2(const T &a, const T &b,
                             unsigned int &which) {
    T result;
    if (a<b) {
      result = b;
      which  = 1;
    }
    else {
      result = a;
      which  = 0;
    }
    return result;
  }

  template<typename T>
  APRIL_CUDA_EXPORT T r_min2(const T &a, const T &b,
                             unsigned int &which) {
    T result;
    if (a<b) {
      result = a;
      which  = 0;
    }
    else {
      result = b;
      which  = 1;
    }
    return result;
  }

  template<typename T>
  APRIL_CUDA_EXPORT T r_add(const T &a, const T &b) {
    return a+b;
  }

  template<typename T>
  APRIL_CUDA_EXPORT T r_mul(const T &a, const T &b) {
    return a*b;
  }

  template<typename T>
  APRIL_CUDA_EXPORT bool r_and(const T &a, const T &b) {
    return a && b;
  }

  template<typename T>
  APRIL_CUDA_EXPORT bool r_or(const T &a, const T &b) {
    return a || b;
  }

  ///////////////////////
  // Curried functions //
  ///////////////////////
  template<typename T> struct m_curried_clamp {
    const T inf, sup;
    m_curried_clamp(const T &inf, const T &sup) : inf(inf), sup(sup) { }
    APRIL_CUDA_EXPORT T operator()(const T &u) const { return m_clamp(u, inf, sup); }
  };

  template<typename T> struct m_curried_pow {
    const T power;
    m_curried_pow(const T &power) : power(power) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) const { return m_pow(a, power); }
  };
  
  template<typename T>
  struct m_curried_mask {
    const T mask_value;
    m_curried_mask(const T &mask_value) : mask_value(mask_value) { }
    APRIL_CUDA_EXPORT T operator()(const T &unit, const T &mask) const {
      return (mask < T(0.5f)) ? mask_value : unit;
    }
  };

  template<typename T> struct m_curried_fill {
    const T value;
    m_curried_fill(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &u) const {
      UNUSED_VARIABLE(u);
      return value;
    }
  };
  
  template<typename T>
  struct m_curried_clamp_der {
    const T inf, sup;
    m_curried_clamp_der(const T &inf, const T &sup) : inf(inf), sup(sup) { }
    APRIL_CUDA_EXPORT T operator()(const T &input, const T &output) const {
      return m_clamp_der(input, output, inf, sup);
    }
  };
  
  template<typename T>
  struct m_curried_lt {
    const T value;
    m_curried_lt(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return m_lt(a, value);
    }
  };

  template<typename T>
  struct m_curried_gt {
    const T value;
    m_curried_gt(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return m_gt(a, value);
    }
  };

  template<typename T>
  struct m_curried_eq {
    const T value;
    m_curried_eq(const T &value) : value(value) {
      if (m_isnan(value)) {
        ERROR_EXIT(128, "For NaN comparison use m_curried_eq_nan\n");
      }
    }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      if (a == value) return T(1.0f);
      else return T(0.0f);
    }
  };

  template<typename T>
  struct m_curried_eq_nan {
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      if (m_isnan(a)) return T(1.0f);
      else return T(0.0f);
    }
  };

  template<typename T>
  struct m_curried_neq {
    const T value;
    m_curried_neq(const T &value) : value(value) {
      if (m_isnan(value)) {
        ERROR_EXIT(128, "For NaN comparison use m_curried_eq_nan\n");
      }
    }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      if (a == value) return T(0.0f);
      else return T(1.0f);
    }
  };

  template<typename T>
  struct m_curried_neq_nan {
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      if (m_isnan(a)) return T(0.0f);
      else return T(1.0f);
    }
  };
  
  template<typename T>
  struct m_curried_add {
    const T value;
    m_curried_add(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return a + value;
    }
  };

  template<typename T>
  struct m_curried_mul {
    const T value;
    m_curried_mul(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return a*value;
    }
  };

  template<typename T>
  struct m_curried_div {
    const T value;
    m_curried_div(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return value / a;
    }
  };
  
  template<typename T>
  struct m_curried_relative_equals {
    const T epsilon;
    m_curried_relative_equals(const T &epsilon) : epsilon(epsilon) { }
    APRIL_CUDA_EXPORT bool operator()(const T &a, const T &b) {
      return m_relative_equals(a, b, epsilon);
    }
  };

#undef SCALAR_MAP_TEMPLATE
#undef SCALAR_STD_CMATH_MAP_TEMPLATE

} // namespace april_math

#endif // CMATH_OVERLOADS_H
