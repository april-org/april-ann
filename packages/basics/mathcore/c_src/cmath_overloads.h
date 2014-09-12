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

/// Constant for values close to zero.
#define NEAR_ZERO             1e-6f
/**
 * @brief A derivative is considered saturated when its abs is greater than this
 * constant.
 */
#define DERIVATIVE_SATURATION 17.0f

// Documentation of AprilMath namespace is in mathcore.h
namespace AprilMath {

  /// Logarithm of NEAR_ZERO for float numbers.
  const float  logf_NZ = logf(NEAR_ZERO);
  /// Logarithm of NEAR_ZERO for double numbers.
  const double log_NZ  = log(NEAR_ZERO);
  
  ///////////////// NUMERIC LIMITS /////////////////
  
  /**
   * @brief A replacement of std::limits implemented to avoid dependencies in
   * C++ std namespace.
   */
  template<typename T>
  class Limits {
  public:
    /// Returns the minimum finite value of a typename T.
    static T min() { return T(); }
    /// Returns the maximum finite value of a typename T.
    static T max() { return T(); }
    /**
     * @brief Machine the difference between 1 and the least value greater
     * than 1 that is representable for typename T.
     */
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

  /**
   * @brief Operations over scalar types are defined as C++ functors in this
   * namespace.
   *
   * This functors allow to implement map or reduce operations defined in
   * AprilMath.
   */
  namespace Functors {
    /// Functor for comparison with a NaN value.
    template<typename T>  
    struct m_isnan {
      /**
       * @brief The operator returns @c true/false if @c v is NaN or not.
       *
       * @note By definition, a NaN is always different of any other value, even
       * another NaN, so this operator identifies it using <tt>return v != v</tt>
       */
      APRIL_CUDA_EXPORT bool operator()(const T &v) const {
        return v != v;
      }
    };
  }
  
  /**
   * @brief Function for instantiation and call of Functors::m_isnan::operator()
   * @see Functors::m_isnan
   */
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
  
  /**
   * @struct Functors::m_abs
   *
   * @brief Overloaded abs operation, currently defined for float, double and
   * ComplexF types.
   *
   * @note Its @c operator() returns a float always.
   */
  /**
   * @fn m_abs
   *
   * @brief Function for instantiation and call of Functors::m_abs::operator()
   * @see Functors::m_abs
   */

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
  
  /**
   * @struct Functors::m_sqrt
   *
   * @brief Overloaded sqrt operation, currently defined for float, double and
   * ComplexF types.
   *
   * @note Its @c operator() returns a float always.
   */
  /**
   * @fn m_sqrt
   *
   * @brief Function for instantiation and call of Functors::m_sqrt::operator()
   * @see Functors::m_sqrt
   */
  
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

  /**
   * @struct Functors::m_log
   *
   * @brief Overloaded log operation, currently defined for float and double
   */
  /**
   * @fn m_log
   *
   * @brief Function for instantiation and call of Functors::m_log::operator()
   * @see Functors::m_log
   */

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
  
  /**
   * @struct Functors::m_log1p
   *
   * @brief Overloaded log1p operation, currently defined for float and double.
   */
  /**
   * @fn m_log1p
   *
   * @brief Function for instantiation and call of Functors::m_log1p::operator()
   * @see Functors::m_log1p
   */
  // log1p overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_log1p, log1p);
  
  /**
   * @struct Functors::m_exp
   *
   * @brief Overloaded exp operation, currently defined for float and double.
   */
  /**
   * @fn m_exp
   *
   * @brief Function for instantiation and call of Functors::m_exp::operator()
   * @see Functors::m_exp
   */

  // exp overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_exp, exp);
  
  /**
   * @struct Functors::m_pow
   *
   * @brief Overloaded pow operation, currently defined for float and double.
   */
  /**
   * @fn m_pow
   *
   * @brief Function for instantiation and call of Functors::m_pow::operator()
   * @see Functors::m_pow
   */
  
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

  /**
   * @struct Functors::m_cos
   *
   * @brief Overloaded cos operation, currently defined for float and double.
   */
  /**
   * @fn m_cos
   *
   * @brief Function for instantiation and call of Functors::m_cos::operator()
   * @see Functors::m_cos
   */
  
  // cos overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_cos, cos);
  
  /**
   * @struct Functors::m_acos
   *
   * @brief Overloaded acos operation, currently defined for float and double.
   */
  /**
   * @fn m_acos
   *
   * @brief Function for instantiation and call of Functors::m_acos::operator()
   * @see Functors::m_acos
   */

  // acos overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_acos, acos);

  /**
   * @struct Functors::m_cosh
   *
   * @brief Overloaded cosh operation, currently defined for float and double.
   */
  /**
   * @fn m_cosh
   *
   * @brief Function for instantiation and call of Functors::m_cosh::operator()
   * @see Functors::m_cosh
   */

  // cosh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_cosh, cosh);

  /**
   * @struct Functors::m_acosh
   *
   * @brief Overloaded acosh operation, currently defined for float and double.
   */
  /**
   * @fn m_acosh
   *
   * @brief Function for instantiation and call of Functors::m_acosh::operator()
   * @see Functors::m_acosh
   */

  // acosh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_acosh, acosh);

  /**
   * @struct Functors::m_sin
   *
   * @brief Overloaded sin operation, currently defined for float and double.
   */
  /**
   * @fn m_sin
   *
   * @brief Function for instantiation and call of Functors::m_sin::operator()
   * @see Functors::m_sin
   */

  // sin overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_sin, sin);

  /**
   * @struct Functors::m_asin
   *
   * @brief Overloaded asin operation, currently defined for float and double.
   */
  /**
   * @fn m_asin
   *
   * @brief Function for instantiation and call of Functors::m_asin::operator()
   * @see Functors::m_asin
   */

  // asin overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_asin, asin);

  /**
   * @struct Functors::m_sinh
   *
   * @brief Overloaded sinh operation, currently defined for float and double.
   */
  /**
   * @fn m_sinh
   *
   * @brief Function for instantiation and call of Functors::m_sinh::operator()
   * @see Functors::m_sinh
   */

  // sinh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_sinh, sinh);

  /**
   * @struct Functors::m_asinh
   *
   * @brief Overloaded asinh operation, currently defined for float and double.
   */
  /**
   * @fn m_asinh
   *
   * @brief Function for instantiation and call of Functors::m_asinh::operator()
   * @see Functors::m_asinh
   */

  // asinh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_asinh, asinh);

  /**
   * @struct Functors::m_tan
   *
   * @brief Overloaded tan operation, currently defined for float and double.
   */
  /**
   * @fn m_tan
   *
   * @brief Function for instantiation and call of Functors::m_tan::operator()
   * @see Functors::m_tan
   */

  // tan overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_tan, tan);

  /**
   * @struct Functors::m_atan
   *
   * @brief Overloaded atan operation, currently defined for float and double.
   */
  /**
   * @fn m_atan
   *
   * @brief Function for insatantiation and call of Functors::m_atan::operator()
   * @see Functors::m_atan
   */
  
  // atan overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_atan, atan);

  /**
   * @struct Functors::m_tanh
   *
   * @brief Overloaded tanh operation, currently defined for float and double.
   */
  /**
   * @fn m_tanh
   *
   * @brief Function for instantiation and call of Functors::m_tanh::operator()
   * @see Functors::m_tanh
   */

  // tanh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_tanh, tanh);

  /**
   * @struct Functors::m_atanh
   *
   * @brief Overloaded atanh operation, currently defined for float and double.
   */
  /**
   * @fn m_atanh
   *
   * @brief Function for instantiation and call of Functors::m_atanh::operator()
   * @see Functors::m_atanh
   */
  
  // atanh overload
  SCALAR_STD_CMATH_MAP_TEMPLATE(m_atanh, atanh);

  //
  namespace Functors {
  
    /// Identity functor, it can be used to copy vectors using map.
    template<typename T>
    struct m_identity {
      /// Returns by value its given parameter.
      APRIL_CUDA_EXPORT T operator()(const T &a) const {
        return a;
      }
    };
    
    /// Implementation of <tt> p log(p) </tt>.
    template<typename T>
    struct m_plogp {
      /**
       * @brief <tt> p log(p) </tt> is simplified to take into account the case
       * <tt> 0 log(0) = 0 </tt>
       */
      APRIL_CUDA_EXPORT T operator()(const T &x) const {
        return ((x) > T(0.0f) || (x) < T(0.0f)) ? (x) * AprilMath::m_log(x) : (x);
      }
    };
    
    /// Sign functor.
    template<typename T>
    struct m_sign {
      /**
       * @brief Returns the sign of the given parameter.
       * 
       * It uses a comparison with T(0.0f) to decide the sign, and returns
       * T(-1.0f) or T(1.0f) depending in the result of previous comparison.
       */
      APRIL_CUDA_EXPORT T operator()(const T &x) const {
        return ((x)<T(0.0f)) ? T(-1.0f) : ( ((x)>T(0.0f)) ? T(1.0f) : T(0.0f) );
      }
    };
    
    /// Complement functor.
    template<typename T>
    struct m_complement {
      /// Returns <tt> T(1.0f) - x </tt>
      APRIL_CUDA_EXPORT T operator()(const T &x) const {
        return (T(1.0f) - (x));
      }
    };
    
    /// Clamp functor.
    template<typename T>
    struct m_clamp {
      /// Returns the clamp of @c x using the range <tt>[min,max]</tt>
      APRIL_CUDA_EXPORT T operator()(const T &x,
                                     const T &min,
                                     const T &max) const {
        return ((x)<min?min:((x)>max?max:x));
      }
    };
    
    /// Sigmoid function.
    template<typename T>
    struct m_sigmoid {
      /// Returns \f$ \displaystyle{ \frac{n}{e^{-v} + 1.0} } \f$ where @c n is numerator param.
      APRIL_CUDA_EXPORT T operator()(const T &numerator,
                                     const T &value) const {
        return (numerator) / (AprilMath::m_exp(-(value))+T(1.0f));
      }
    };

    /// Rectified Linear Unit (ReLU) function.
    template<typename T>
    struct m_relu {
      /// Returns <tt> max(0, value) </tt>
      APRIL_CUDA_EXPORT T operator()(const T &value) const {
        return (value > T(0.0f)) ? (value) : T(0.0f);
      }
    };
    
    /// Less than comparison.
    template<typename T>
    struct m_lt {
      /// Returns @c T(1.0f) or @c T(0.0f) depending in @c a<b
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        if (a < b) return T(1.0f);
        else return T(0.0f);
      }
    };
    
    /// Greater than comparison.
    template<typename T>
    struct m_gt {
      /// Returns @c T(1.0f) or @c T(0.0f) depending in @c b<a
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        if (b < a) return T(1.0f);
        else return T(0.0f);
      }
    };
    
    /// Equals comparison.
    template<typename T>
    struct m_eq {
      /// Returns @c T(1.0f) or @c T(0.0f) depending in @c b=a
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
    
    /// Equals comparison using a percentage threshold.
    template<typename T>
    struct m_relative_equals {
      /**
       * @brief Compares @c a=b using @c TH as percentage of relative error.
       * @returns True or false.
       */
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

    /// Multiplication map operation.
    template<typename T>
    struct m_mul {
      /// Returns \f$ a*b \f$
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return a*b;
      }
    };

    /// Addition map operation.
    template<typename T>
    struct m_add {
      /// Returns \f$ a+b \f$
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return a+b;
      }
    };

    /// Division map operation.
    template<typename T>
    struct m_div {
      /// Returns \f$ a/b \f$
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return a/b;
      }
    };

    /// Maximum map operation.
    template<typename T>
    struct m_max {
      /// Returns \f$ max(a,b) \f$
      APRIL_CUDA_EXPORT T operator()(const T &a, const T &b) const {
        return (a<b)?b:a;
      }
    };

    /// Minimum map operation.
    template<typename T>
    struct m_min {
      /// Returns \f$ min(a,b) \f$
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
  /// @see Functors::m_div
  template<typename T> APRIL_CUDA_EXPORT
  T m_div(const T &a, const T &b) { return Functors::m_div<T>()(a,b); }
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
    
    template<typename T, typename O, typename MAP_OP, typename RED_OP>
    struct r_map1 {
      MAP_OP map_functor;
      RED_OP red_functor;
      r_map1(const MAP_OP &map_functor, const RED_OP &red_functor) :
        map_functor(map_functor), red_functor(red_functor) { }
      APRIL_CUDA_EXPORT void operator()(O &acc, const T &b) const {
        red_functor(acc, map_functor(b));
      }
    };

    template<typename T1, typename T2, typename O, typename MAP_OP, typename RED_OP>
    struct r_map2 {
      MAP_OP map_functor;
      RED_OP red_functor;
      r_map2(const MAP_OP &map_functor, const RED_OP &red_functor) :
        map_functor(map_functor), red_functor(red_functor) { }
      APRIL_CUDA_EXPORT void operator()(O &acc, const T1 &b, const T2 &c) const {
        red_functor(acc, map_functor(b,c));
      }
    };
    
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
          which_acc = b_idx;
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
          which_acc = b_idx;
        }
      }
    };

    /**
     * @brief Reduction for @c acc+=v
     * @note Uses the C++ @c operator+=
     */
    template<typename T, typename O>
    struct r_add {
      APRIL_CUDA_EXPORT void operator()(O &acc, const T &v) const {
        acc += v;
      }
    };
    
    /**
     * @brief Reduction for @c acc*=v
     * @note Uses the C++ @c operator*=
     */
    template<typename T, typename O>
    struct r_mul {
      APRIL_CUDA_EXPORT void operator()(O &acc, const T &v) const {
        acc *= v;
      }
    };

    /**
     * @brief Reduction for @c acc/=v
     * @note Uses the C++ @c operator/=
     */
    template<typename T, typename O>
    struct r_div {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &v) const {
        acc /= v;
      }
    };
    
    /**
     * @brief Reduction for @c acc=acc&&v
     * @note Uses the C++ @c operator&&
     */
    template<typename T>
    struct r_and {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b) const {
        acc = acc && b;
      }
    };

    /**
     * @brief Reduction for @c acc=acc||v
     * @note Uses the C++ @c operator||
     */
    template<typename T>
    struct r_or {
      APRIL_CUDA_EXPORT void operator()(T &acc, const T &b) const {
        acc = acc || b;
      }
    };
  } // namespace Functors

  /// @see Functors::r_map1
  template<typename T, typename O, typename MAP_OP, typename RED_OP>
  Functors::r_map1<T,O,MAP_OP,RED_OP> make_r_map1(const MAP_OP &map_functor,
                                                  const RED_OP &red_functor) {
    return Functors::r_map1<T,O,MAP_OP,RED_OP>(map_functor, red_functor);
  }
  
  /// @see Functors::r_map2
  template<typename T1, typename T2, typename O,
           typename MAP_OP, typename RED_OP>
  Functors::r_map2<T1,T2,O,MAP_OP,RED_OP> make_r_map2(const MAP_OP &map_functor,
                                                      const RED_OP &red_functor) {
    return Functors::r_map2<T1,T2,O,MAP_OP,RED_OP>(map_functor, red_functor);
  }
  
  /// @see Functors::r_max
  template<typename T> APRIL_CUDA_EXPORT
  void r_max(T &acc, const T &b) { Functors::r_max<T>()(acc,b); }
  /// @see Functors::r_min
  template<typename T> APRIL_CUDA_EXPORT
  void r_min(T &acc, const T &b) { Functors::r_min<T>()(acc,b); }
  /// @see Functors::r_max2
  template<typename T> APRIL_CUDA_EXPORT
  void r_max2(T &acc, const T &b, int32_t &c, const int32_t d) { Functors::r_max2<T>()(acc,b,c,d); }
  /// @see Functors::r_min2
  template<typename T> APRIL_CUDA_EXPORT
  void r_min2(T &acc, const T b, int32_t &c, const int32_t d) { Functors::r_min2<T>()(acc,b,c,d); }
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
