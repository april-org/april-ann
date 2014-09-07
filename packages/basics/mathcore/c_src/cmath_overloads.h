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

/**
 * @brief The namespace AprilMath contains operations over different scalar types by
 * using templatized functions and C++ functors.
 *
 * All of this operations are exported to CUDA and can be used safely in
 * functions implemented to run in GPU device or CPU host.
 */
namespace AprilMath {
  
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
  
  template<> char Limits<char>::min();
  template<> char Limits<char>::max();

  template<> int32_t Limits<int32_t>::min();
  template<> int32_t Limits<int32_t>::max();
  
  template<> float Limits<float>::min();
  template<> float Limits<float>::max();
  template<> float Limits<float>::epsilon();
  
  template<> double Limits<double>::min();
  template<> double Limits<double>::max();
  template<> double Limits<double>::epsilon();
  
  template<> ComplexF Limits<ComplexF>::min();
  template<> ComplexF Limits<ComplexF>::max();
  template<> ComplexF Limits<ComplexF>::epsilon();

  ///////////////// NAN CHECK /////////////////

  namespace Functors {
    template<typename T>  
    struct m_isnan {
      APRIL_CUDA_EXPORT bool operator()(const T &v) const {
        return v != v; // by definition, a NAN is always different of any other
        // value, even another NAN
      }
    };
  }
  template<typename T>
  APRIL_CUDA_EXPORT bool m_isnan(const T &v) { return Functors::m_isnan<T>()(v); }
  
  //////////////// MATH SCALAR MAP FUNCTIONS ////////////////
  
#define SCALAR_MAP_TEMPLATE(NAME, T_IN, T_OUT)                          \
  namespace Functors {                                                  \
    template<typename T>                                                \
    struct NAME {                                                       \
      APRIL_CUDA_EXPORT T_OUT operator()(const T_IN &v) const {         \
        UNUSED_VARIABLE(v);                                             \
        APRIL_CUDA_ERROR_EXIT(128, "NOT IMPLEMENTED\n");                \
        return T_OUT();                                                 \
      }                                                                 \
    };                                                                  \
  }                                                                     \
  template<typename T>                                                  \
  APRIL_CUDA_EXPORT T_OUT NAME(const T_IN &v) { return Functors::NAME<T>()(v); }
  
#define SCALAR_STD_CMATH_MAP_TEMPLATE(NAME,CFUNC)                       \
  SCALAR_MAP_TEMPLATE(NAME, T, T);                                      \
  namespace Functors {                                                  \
    template<> struct NAME<float> {                                     \
      APRIL_CUDA_EXPORT float operator()(const float &v) const { return CFUNC##f(v); } \
    };                                                                  \
    template<> struct NAME<double> {                                    \
      APRIL_CUDA_EXPORT double operator()(const double &v) const { return CFUNC(v); } \
    };                                                                  \
  }
  
  // abs overload
  SCALAR_MAP_TEMPLATE(m_abs, T, float);
  namespace Functors {
    template<> struct m_abs<float> {
      APRIL_CUDA_EXPORT float operator()(const float &v) const { return fabsf(v); }
    };
    template<> struct m_abs<double> {
      APRIL_CUDA_EXPORT float operator()(const double &v) const { return fabs(v); }
    };
    template<> struct m_abs<ComplexF> {
      APRIL_CUDA_EXPORT float operator()(const ComplexF &v) const { return v.abs(); }
    };
  }
  
  // sqrt overload
  SCALAR_MAP_TEMPLATE(m_sqrt, T, float);
  namespace Functors {
    template<> struct m_sqrt<float> {
      APRIL_CUDA_EXPORT float operator()(const float &v) const { return sqrtf(v); }
    };
    template<> struct m_sqrt<double> {
      APRIL_CUDA_EXPORT float operator()(const double &v) const { return sqrt(v); }
    };
    template<> struct m_sqrt<ComplexF> {
      APRIL_CUDA_EXPORT float operator()(const ComplexF &v) const { return v.sqrtc(); }
    };
  }

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
  namespace Functors {
    template<typename T>
    struct m_pow {
      APRIL_CUDA_EXPORT T operator()(const T &v, const T &p) const {
        UNUSED_VARIABLE(v);
        UNUSED_VARIABLE(p);
        APRIL_CUDA_ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        return T();
      }
    };
    template<> struct m_pow<float> {
      APRIL_CUDA_EXPORT float operator()(const float &v, const float &p) const {
        return powf(v, p);
      }
    };
    template<> struct m_pow<double> {
      APRIL_CUDA_EXPORT double operator()(const double &v, const double &p) const {
        return pow(v, p);
      }
    };
  }
  template<typename T> APRIL_CUDA_EXPORT
  T m_pow(const T &a, const T &b) { return Functors::m_pow<T>()(a,b); }

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
  namespace Functors {
    
    template<typename T>
    struct m_identity {
      APRIL_CUDA_EXPORT T operator()(const T &a) const {
        return a;
      }
    };
    
    template<typename T>
    struct m_plogp {
      APRIL_CUDA_EXPORT T operator()(const T &x) const {
        return ((x) > T(0.0f) || (x) < T(0.0f)) ? (x) * AprilMath::m_log(x) : (x);
      }
    };
    
    template<typename T>
    struct m_sign {
      APRIL_CUDA_EXPORT T operator()(const T &x) const {
        return ((x)<T(0.0f)) ? T(-1.0f) : ( ((x)>T(0.0f)) ? T(1.0f) : T(0.0f) );
      }
    };
    
    template<typename T>
    struct m_complement {
      APRIL_CUDA_EXPORT T operator()(const T &x) const {
        return (T(1.0f) - (x));
      }
    };
    
    template<typename T>
    struct m_clamp {
      APRIL_CUDA_EXPORT T operator()(const T &x,
                                     const T &min,
                                     const T &max) const {
        return ((x)<min?min:((x)>max?max:x));
      }
    };
    
    template<typename T>
    struct m_sigmoid {
      APRIL_CUDA_EXPORT T operator()(const T &numerator,
                                     const T &value) const {
        return (numerator) / (AprilMath::m_exp(-(value))+T(1.0f));
      }
    };

    template<typename T>
    struct m_relu {
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        return (value > T(0.0f)) ? (value) : T(0.0f);
      }
    };
    
    template<typename T>
    struct m_lt {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        if (a < b) return T(1.0f);
        else return T(0.0f);
      }
    };
    
    template<typename T>
    struct m_gt {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        if (b < a) return T(1.0f);
        else return T(0.0f);
      }
    };
    
    template<typename T>
    struct m_eq {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        if (AprilMath::m_isnan(a)) {
          if (AprilMath::m_isnan(b)) return T(1.0f);
          else return T(0.0f);
        }
        else {
          if (a == b) return T(1.0f);
          else return T(0.0f);
        }
      }
    };
    
    template<typename T>
    struct m_relative_equals {
      APRIL_CUDA_EXPORT bool operator()(const T &a,
                                        const T &b,
                                        const float &TH) const {
        if (AprilMath::m_isnan(a) && AprilMath::m_isnan(b)) {
          return true;
        }
        else if (a == b) {
          return true;
        }
        else {
          const float ZERO = 1e-03;
          float a_abs = AprilMath::m_abs(a);
          float b_abs = AprilMath::m_abs(b);
          if (a_abs < ZERO || b_abs < ZERO) {
            if (a_abs < ZERO && b_abs < ZERO) {
              return true;
            }
            else if (AprilMath::m_abs(a_abs-ZERO)/(a_abs+ZERO) > TH ||
                     AprilMath::m_abs(b_abs-ZERO)/(b_abs+ZERO) > TH) {
              return false;
            }
            else {
              return true;
            }
          }
          else {
            float diff = 2.0f * ( AprilMath::m_abs(a - b) / (a_abs + b_abs) );
            if (diff < TH) {
              return true;
            }
            else {
              return false;
            }
          }
        }
      }
    };

    template<typename T>
    struct m_mul {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return a*b;
      }
    };

    template<typename T>
    struct m_add {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return a+b;
      }
    };

    template<typename T>
    struct m_max {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return (a<b)?b:a;
      }
    };

    template<typename T>
    struct m_min {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return (a<b)?a:b;
      }
    };
  } // namespace Functors
  
  /// @see Functors::m_identity
  template<typename T> APRIL_CUDA_EXPORT
  T m_identity(const T &a) { return Functors::m_identity<T>()(a); }
  /// @see Functors::m_plogp
  template<typename T> APRIL_CUDA_EXPORT
  T m_plogp(const T &a) { return Functors::m_plogp<T>()(a); }
  /// @see Functors::m_sign
  template<typename T> APRIL_CUDA_EXPORT
  T m_sign(const T &a) { return Functors::m_sign<T>()(a); }
  /// @see Functors::m_complement
  template<typename T> APRIL_CUDA_EXPORT
  T m_complement(const T &a) { return Functors::m_complement<T>()(a); }
  /// @see Functors::m_clamp
  template<typename T> APRIL_CUDA_EXPORT
  T m_clamp(const T &a, const T &b, const T &c) { return Functors::m_clamp<T>()(a,b,c); }
  /// @see Functors::m_sigmoid
  template<typename T> APRIL_CUDA_EXPORT
  T m_sigmoid(const T &a, const T &b) { return Functors::m_sigmoid<T>()(a,b); }
  /// @see Functors::m_relu
  template<typename T> APRIL_CUDA_EXPORT
  T m_relu(const T &a) { return Functors::m_relu<T>()(a); }
  /// @see Functors::m_lt
  template<typename T> APRIL_CUDA_EXPORT
  T m_lt(const T &a, const T &b) { return Functors::m_lt<T>()(a,b); }
  /// @see Functors::m_gt
  template<typename T> APRIL_CUDA_EXPORT
  T m_gt(const T &a, const T &b) { return Functors::m_gt<T>()(a,b); }
  /// @see Functors::m_eq
  template<typename T> APRIL_CUDA_EXPORT
  T m_eq(const T &a, const T &b) { return Functors::m_eq<T>()(a,b); }
  /// @see Functors::m_relative_equals
  template<typename T> APRIL_CUDA_EXPORT
  bool m_relative_equals(const T &a, const T &b, const float &c) { return Functors::m_relative_equals<T>()(a,b,c); }
  /// @see Functors::m_mul
  template<typename T> APRIL_CUDA_EXPORT
  T m_mul(const T &a, const T &b) { return Functors::m_mul<T>()(a,b); }
  /// @see Functors::m_add
  template<typename T> APRIL_CUDA_EXPORT
  T m_add(const T &a, const T &b) { return Functors::m_add<T>()(a,b); }
  /// @see Functors::m_min
  template<typename T> APRIL_CUDA_EXPORT
  T m_min(const T &a, const T &b) { return Functors::m_min<T>()(a,b); }
  /// @see Functors::m_max
  template<typename T> APRIL_CUDA_EXPORT
  T m_max(const T &a, const T &b) { return Functors::m_max<T>()(a,b); }

  
  namespace Functors {
    
    template<typename T>
    struct m_neq {
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) {
        if (AprilMath::m_eq(a,b) == T(1.0f)) return T(0.0f);
        else return T(1.0f);
      }
    };
    
    template<typename T>
    struct m_logistic {
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        return AprilMath::m_sigmoid(T(1.0f), value);
      }
    };
    
    template<typename T>
    struct m_antisym_logistic {
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        return AprilMath::m_sigmoid(T(2.0f), value) - T(1.0f);
      }
    };
    
    template<typename T>
    struct m_log_logistic {
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        // The value of -log1p(exp(x)) when X is negative and large, is
        // approximately X
        return ( (value)<T(-10.0f)) ? (value) : (-AprilMath::m_log1p(AprilMath::m_exp(-(value))));
      }
    };
    
    template<typename T>
    struct m_softsign {
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        return value / (T(1.0f) + AprilMath::m_abs(value));
      }
    };
    
    template<typename T>
    struct m_softplus {
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        return AprilMath::m_log1p(AprilMath::m_exp(value));
      }
    };
  } // namespace Functors
  
  /// @see Functors::m_neq
  template<typename T> APRIL_CUDA_EXPORT
  T m_neq(const T &a, const T &b) { return Functors::m_neq<T>()(a,b); }
  /// @see Functors::m_logistic
  template<typename T> APRIL_CUDA_EXPORT
  T m_logistic(const T &a) { return Functors::m_logistic<T>()(a); }
  /// @see Functors::m_log_logistic
  template<typename T> APRIL_CUDA_EXPORT
  T m_log_logistic(const T &a) { return Functors::m_log_logistic<T>()(a); }
  /// @see Functors::m_antisym_logistic
  template<typename T> APRIL_CUDA_EXPORT
  T m_antisym_logistic(const T &a) { return Functors::m_antisym_logistic<T>()(a); }
  /// @see Functors::m_softsign
  template<typename T> APRIL_CUDA_EXPORT
  T m_softsign(const T &a) { return Functors::m_softsign<T>()(a); }
  /// @see Functors::m_softplus
  template<typename T> APRIL_CUDA_EXPORT
  T m_softplus(const T &a) { return Functors::m_softplus<T>()(a); }
  
  // DERIVATIVES
  namespace Functors {  
    
    template<typename T>
    struct m_logistic_der {
      APRIL_CUDA_EXPORT T operator()(const T &after_actf) const {
        float value = AprilMath::m_clamp(after_actf, NEAR_ZERO, T(1.0f) - NEAR_ZERO);
        return value * (T(1.0f) - value);
      }
    };
    
    template<typename T>
    struct m_antisym_logistic_der {
      APRIL_CUDA_EXPORT T operator()(const T &after_actf) const {
        T value = AprilMath::m_clamp(after_actf, T(-1.0f) + NEAR_ZERO, T(1.0f) - NEAR_ZERO);
        return T(0.5f) * (T(1.0f) - (value*value));
      }
    };
    
    template<typename T>
    struct m_softsign_der {
      APRIL_CUDA_EXPORT T operator()(const T &after_actf) const {
        T value = AprilMath::m_clamp(after_actf, T(-1.0f) + NEAR_ZERO, T(1.0f) - NEAR_ZERO);
        T aux   = T(1.0f) + AprilMath::m_abs(value);
        return T(1.0f) / (aux * aux);
      }
    };
    
    template<typename T>
    struct m_softplus_der {
      APRIL_CUDA_EXPORT T operator()(const T &before_actf) const {
        T value = AprilMath::m_logistic(before_actf);
        return value;
      }
    };
    
    template<typename T>
    struct m_relu_der {
      APRIL_CUDA_EXPORT T operator()(const T &before_actf) const {
        return (before_actf > T(0.0f)) ? T(1.0f) : T(0.0f);
      }
    };
    
    template<typename T>
    struct m_clamp_der {
      APRIL_CUDA_EXPORT T operator()(const T &before_actf,
                                     const T &inf,
                                     const T &sup) const {
        return (before_actf < inf || before_actf > sup) ? T(0.0f) : T(1.0f);
      }
    };
  } // namespace Functors
  
  /// @see Functors::m_logistic_der
  template<typename T> APRIL_CUDA_EXPORT
  T m_logistic_der(const T &a) { return Functors::m_logistic_der<T>()(a); }
  /// @see Functors::m_antisym_logistic_der
  template<typename T> APRIL_CUDA_EXPORT
  T m_antisym_logistic_der(const T &a) { return Functors::m_antisym_logistic_der<T>()(a); }
  /// @see Functors::m_softsign_der
  template<typename T> APRIL_CUDA_EXPORT
  T m_softsign_der(const T &a) { return Functors::m_softsign_der<T>()(a); }
  /// @see Functors::m_softplus_der
  template<typename T> APRIL_CUDA_EXPORT
  T m_softplus_der(const T &a) { return Functors::m_softplus_der<T>()(a); }
  /// @see Functors::m_relu_der
  template<typename T> APRIL_CUDA_EXPORT
  T m_relu_der(const T &a) { return Functors::m_relu_der<T>()(a); }
  /// @see Functors::m_clamp_der
  template<typename T> APRIL_CUDA_EXPORT
  T m_clamp_der(const T &a, const T &b, const T &c) { return Functors::m_clamp_der<T>()(a,b,c); }
  
  //////////////// MATH SCALAR REDUCE FUNCTIONS ////////////////
  namespace Functors {
    
    template<typename T>
    struct r_max {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b) const {
        if (acc<b) acc = b;
      }
    };

    template<typename T>
    struct r_min {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b) const {
        if (!(acc < b)) acc = b;
      }
    };
    
    template<typename T>
    struct r_max2 {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b,
                                        int32_t &which_acc,
                                        const int32_t b_idx) const {
        if (acc<b) {
          acc = b;
          which_acc = b_idx+1; // +1 because Lua starts at 1
        }
      }
    };

    template<typename T>
    struct r_min2 {
      APRIL_CUDA_EXPORT void operator()(T & acc, const T &b,
                                        int32_t &which_acc,
                                        const int32_t b_idx) const {
        if (!(acc<b)) {
          acc = b;
          which_acc = b_idx+1; // +1 because Lua starts at 1
        }
      }
    };

    template<typename T, typename O>
    struct r_add {
      APRIL_CUDA_EXPORT void operator()(O &acc, const T &v) const {
        acc += v;
      }
    };
    
    template<typename T, typename O>
    struct r_mul {
      APRIL_CUDA_EXPORT void operator()(O &acc, const T &v) const {
        acc *= v;
      }
    };

    template<typename T, typename O>
    struct r_div {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &v) const {
        acc /= v;
      }
    };
    
    template<typename T>
    struct r_and {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b) const {
        acc = acc && b;
      }
    };

    template<typename T>
    struct r_or {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b) const {
        acc = acc || b;
      }
    };
  } // namespace Functors
  
  /// @see Functors::r_max
  template<typename T> APRIL_CUDA_EXPORT
  void r_max(T &acc, const T &a, const T &b) { Functors::r_max<T>()(acc,a,b); }
  /// @see Functors::r_min
  template<typename T> APRIL_CUDA_EXPORT
  void r_min(T &acc, const T &a, const T &b) { Functors::r_min<T>()(acc,a,b); }
  /// @see Functors::r_max2
  template<typename T> APRIL_CUDA_EXPORT
  void r_max2(T &acc, const T &a, const T &b, int32_t &c, const int32_t d) { Functors::r_max2<T>()(acc,a,b,c,d); }
  /// @see Functors::r_min2
  template<typename T> APRIL_CUDA_EXPORT
  void r_min2(const T &a, const T &b, int32_t &c, const int32_t d) { Functors::r_min2<T>()(a,b,c,d); }
  /// @see Functors::r_add
  template<typename T, typename O> APRIL_CUDA_EXPORT
  void r_add(O &a, const T &b) { Functors::r_add<T,O>()(a,b); }
  /// @see Functors::r_mul
  template<typename T,typename O> APRIL_CUDA_EXPORT
  void r_mul(O &a, const T &b) { Functors::r_mul<T,O>()(a,b); }
  /// @see Functors::r_div
  template<typename T,typename O> APRIL_CUDA_EXPORT
  void r_div(O &a, const T &b) { return Functors::r_div<T,O>()(a,b); }
  /// @see Functors::r_and
  template<typename T> APRIL_CUDA_EXPORT
  void r_and(T &a, const T &b) { Functors::r_and<T>()(a,b); }
  /// @see Functors::r_or
  template<typename T> APRIL_CUDA_EXPORT
  void r_or(T &a, const T &b) { Functors::r_or<T>()(a,b); }

  ///////////////////////
  // Curried functions //
  ///////////////////////
  
  template<typename T> struct m_curried_clamp {
    const T inf, sup;
    m_curried_clamp(const T &inf, const T &sup) : inf(inf), sup(sup) { }
    APRIL_CUDA_EXPORT T operator()(const T &u) const { return AprilMath::m_clamp(u, inf, sup); }
  };

  template<typename T> struct m_curried_pow {
    const T power;
    m_curried_pow(const T &power) : power(power) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) const { return AprilMath::m_pow(a, power); }
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
    APRIL_CUDA_EXPORT T operator()(const T &before_actf) const {
      return AprilMath::m_clamp_der(before_actf, inf, sup);
    }
  };
  
  template<typename T>
  struct m_curried_lt {
    const T value;
    m_curried_lt(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return AprilMath::m_lt(a, value);
    }
  };

  template<typename T>
  struct m_curried_gt {
    const T value;
    m_curried_gt(const T &value) : value(value) { }
    APRIL_CUDA_EXPORT T operator()(const T &a) {
      return AprilMath::m_gt(a, value);
    }
  };

  template<typename T>
  struct m_curried_eq {
    const T value;
    m_curried_eq(const T &value) : value(value) {
      if (AprilMath::m_isnan(value)) {
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
      if (AprilMath::m_isnan(a)) return T(1.0f);
      else return T(0.0f);
    }
  };

  template<typename T>
  struct m_curried_neq {
    const T value;
    m_curried_neq(const T &value) : value(value) {
      if (AprilMath::m_isnan(value)) {
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
      if (AprilMath::m_isnan(a)) return T(0.0f);
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
      return value/a;
    }
  };
  
  template<typename T>
  struct m_curried_relative_equals {
    const float epsilon;
    m_curried_relative_equals(const float &epsilon) : epsilon(epsilon) { }
    APRIL_CUDA_EXPORT bool operator()(const T &a, const T &b) {
      return AprilMath::m_relative_equals(a, b, epsilon);
    }
  };

#undef SCALAR_MAP_TEMPLATE
#undef SCALAR_STD_CMATH_MAP_TEMPLATE

} // namespace AprilMath

#endif // CMATH_OVERLOADS_H
