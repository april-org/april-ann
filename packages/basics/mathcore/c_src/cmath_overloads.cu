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
#include <climits>

#include "cmath_overloads.h"

#ifndef INT8_MAX
#define INT8_MAX 0x7f
#endif
#ifndef INT8_MIN
#define INT8_MIN (-INT8_MAX - 1)
#endif
#ifndef UINT8_MAX
#define UINT8_MAX 0xffU
#endif  
#ifndef INT16_MAX
#define INT16_MAX 0x7fff
#endif
#ifndef INT16_MIN
#define INT16_MIN (-INT16_MAX - 1)
#endif
#ifndef UINT16_MAX
#define UINT16_MAX 0xffffU
#endif
#ifndef INT32_MAX
#define INT32_MAX 0x7fffffffL
#endif
#ifndef INT32_MIN
#define INT32_MIN (-INT32_MAX - 1L)
#endif
#ifndef UINT32_MAX
#define UINT32_MAX 0xffffffffUL
#endif
#ifndef INT64_MAX
#define INT64_MAX 0x7fffffffffffffffLL
#endif
#ifndef INT64_MIN
#define INT64_MIN (-INT64_MAX - 1LL)
#endif
#ifndef UINT64_MAX
#define UINT64_MAX 0xffffffffffffffffULL
#endif

namespace AprilMath {
  
  template<> char Limits<char>::lowest() { return CHAR_MIN; }
  template<> char Limits<char>::min() { return CHAR_MIN; }
  template<> char Limits<char>::max() { return CHAR_MAX; }

  template<> int8_t Limits<int8_t>::lowest() { return INT8_MIN; }
  template<> int8_t Limits<int8_t>::min() { return INT8_MIN; }
  template<> int8_t Limits<int8_t>::max() { return INT8_MAX; }

  template<> uint8_t Limits<uint8_t>::lowest() { return 0u; }
  template<> uint8_t Limits<uint8_t>::min() { return 0u; }
  template<> uint8_t Limits<uint8_t>::max() { return UINT8_MAX; }

  template<> int16_t Limits<int16_t>::lowest() { return INT16_MIN; }
  template<> int16_t Limits<int16_t>::min() { return INT16_MIN; }
  template<> int16_t Limits<int16_t>::max() { return INT16_MAX; }

  template<> uint16_t Limits<uint16_t>::lowest() { return 0u; }
  template<> uint16_t Limits<uint16_t>::min() { return 0u; }
  template<> uint16_t Limits<uint16_t>::max() { return UINT16_MAX; }

  template<> int32_t Limits<int32_t>::lowest() { return INT32_MIN; }
  template<> int32_t Limits<int32_t>::min() { return INT32_MIN; }
  template<> int32_t Limits<int32_t>::max() { return INT32_MAX; }

  template<> uint32_t Limits<uint32_t>::lowest() { return 0u; }
  template<> uint32_t Limits<uint32_t>::min() { return 0u; }
  template<> uint32_t Limits<uint32_t>::max() { return UINT32_MAX; }
  
  template<> int64_t Limits<int64_t>::lowest() { return INT64_MIN; }
  template<> int64_t Limits<int64_t>::min() { return INT64_MIN; }
  template<> int64_t Limits<int64_t>::max() { return INT64_MAX; }

  template<> uint64_t Limits<uint64_t>::lowest() { return 0u; }
  template<> uint64_t Limits<uint64_t>::min() { return 0u; }
  template<> uint64_t Limits<uint64_t>::max() { return UINT64_MAX; }
  
  template<> float Limits<float>::lowest() { return -FLT_MAX; }
  template<> float Limits<float>::min() { return FLT_MIN; }
  template<> float Limits<float>::max() { return FLT_MAX; }
  template<> float Limits<float>::epsilon() { return FLT_EPSILON; }
  template<> bool Limits<float>::hasInfinity() { return true; }
  template<> float Limits<float>::infinity() { return HUGE_VALF; }
  template<> float Limits<float>::quiet_NaN() { return float(NAN); }

  template<> double Limits<double>::lowest() { return -DBL_MAX; }
  template<> double Limits<double>::min() { return DBL_MIN; }
  template<> double Limits<double>::max() { return DBL_MAX; }
  template<> double Limits<double>::epsilon() { return DBL_EPSILON; }
  template<> bool Limits<double>::hasInfinity() { return true; }
  template<> double Limits<double>::infinity() { return HUGE_VAL; }
  template<> double Limits<double>::quiet_NaN() { return double(NAN); }
  
  template<> ComplexF Limits<ComplexF>::lowest() {
    return ComplexF(Limits<float>::lowest(),Limits<float>::lowest());
  }
  template<> ComplexF Limits<ComplexF>::min() {
    return ComplexF(Limits<float>::min(),Limits<float>::min());
  }
  template<> ComplexF Limits<ComplexF>::max() {
    return ComplexF(Limits<float>::max(),Limits<float>::max());
  }
  template<> ComplexF Limits<ComplexF>::epsilon() {
    return ComplexF(Limits<float>::epsilon(),Limits<float>::epsilon());
  }
  template<> bool Limits<ComplexF>::hasInfinity() { return true; }
  template<> ComplexF Limits<ComplexF>::infinity() {
    return ComplexF(Limits<float>::infinity(),Limits<float>::infinity());
  }
  template<> ComplexF Limits<ComplexF>::quiet_NaN() {
    return ComplexF(Limits<float>::quiet_NaN(), Limits<float>::quiet_NaN());
  }
#ifndef USE_CUDA
  template<> ComplexF Limits<ComplexF>::zero() {
    return ComplexF::zero_zero();
  }
  template<> ComplexF Limits<ComplexF>::one() {
    return ComplexF::one_zero();
  }
#endif
  
} // namespace AprilMath
