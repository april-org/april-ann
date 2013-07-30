/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef LOGBASE_ASSERT_H
#define LOGBASE_ASSERT_H

#include "april_assert.h"
#include <cmath>
#include <cstdio>

// Esta clase representa valores numéricos en base logarítmica.
// Por tanto, sólo tiene sentido para representar números >0
// aunque el 0 se aproxima por un valor muy cercano.

class log_double; // forward declaration

class log_float {
  float raw_value;
  // nota: el minimo numero float no denormal es 1.2e-38
  static const float floatnearzero    = 1e-37;
  static const float rawscorenearzero = -99;
  static const float rawscorezero     = -1e12;
  static const float rawscoreone      = 0;
 public:
  log_float(float x) : raw_value(x) {}
  log_float() : raw_value(rawscorezero) {}
  static log_float zero() { return log_float(rawscorezero); }
  static log_float one()  { return log_float(rawscoreone); }
  static log_float from_float(float value) { 
    return log_float((value < floatnearzero) ? 
		      rawscorezero : ::log(value));
  }
  float to_float() const {
    return expf(raw_value);
  }
  double to_double() const {
    return exp((double)raw_value);
  }
  float log() const {
    return raw_value;
  }

  log_float raise_to(float exponent) {
    float aux = raw_value*exponent;
    april_assert(aux >= rawscorezero);
    return log_float(aux);
  }

  // arithmetic operations

  friend log_float operator + (const log_float &x,
			       const log_float &y);

  friend log_float operator * (const log_float &x,
			       const log_float &y);

  friend log_float operator / (const log_float &x,
			       const log_float &y);

  friend log_float operator - (const log_float &x,
			       const log_float &y);

  log_float operator += (const log_float &y) {
    *this = *this + y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_float operator += (const log_double &y);

  log_float operator *= (const log_float &y) {
    *this = *this * y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_float operator *= (const log_double &y);

  log_float operator /= (const log_float &y) {
    *this = *this / y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_float operator /= (const log_double &y);

  log_float operator -= (const log_float &y) {
    *this = *this - y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_float operator -= (const log_double &y);

  // comparison operations
  friend bool operator == (const log_float &x,
			   const log_float &y);
  
  friend bool operator != (const log_float &x,
			   const log_float &y);
  
  friend bool operator >= (const log_float &x,
			   const log_float &y);
  
  friend bool operator <= (const log_float &x,
			   const log_float &y);
  
  friend bool operator > (const log_float &x,
			  const log_float &y);
  
  friend bool operator < (const log_float &x,
			  const log_float &y);
  
  // conversion
  operator float() const {
    return raw_value;
  }
  operator double() const {
    return (double)raw_value;
  }
  operator unsigned int() const {
    return (unsigned int)(raw_value);
  }

  // assignment
  log_float &operator = (float &value) {
    raw_value = value;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_float &operator = (double &value) {
    raw_value = value;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_float &operator = (const log_double &value);

};

class log_double {
  friend class log_float;
  double raw_value;
  // nota: el minimo numero float no denormal es 1.2e-38
  static const double doublenearzero   = 1e-37;
  static const double rawscorenearzero = -85;
  static const double rawscorezero     = -1e8;
  static const double rawscoreone      = 0;
 public:
  log_double(double x) : raw_value(x) {}
  log_double(const log_float& f): raw_value(double(f)) {}
  log_double() : raw_value(rawscorezero) {}
  static log_double from_double(double value) { 
    return log_double((value < doublenearzero) ? 
		      rawscorezero : ::log(value));
  }
  static log_double zero() { return log_double(rawscorezero); }
  static log_double one()  { return log_double(rawscoreone); }
  double to_double() const {
    return exp(raw_value);
  }
  float to_float() const {
    return expf((float)raw_value);
  }
  double log() const {
    return raw_value;
  }

  // arithmetic operations

  log_double raise_to(double exponent) {
    double aux = raw_value*exponent;
    april_assert(aux >= rawscorezero);
    return log_double(aux);
  }

  friend log_double operator + (const log_double &x,
				const log_double &y);
  
  friend log_double operator * (const log_double &x,
				const log_double &y);
  
  friend log_double operator / (const log_double &x,
				const log_double &y);
  
  friend log_double operator - (const log_double &x,
				const log_double &y);
  
  log_double operator += (const log_double &y) {
    *this = *this * y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_double operator *= (const log_double &y) {
    *this = *this * y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_double operator /= (const log_double &y) {
    *this = *this / y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_double operator -= (const log_double &y) {
    *this = *this - y;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  
  // comparison operations
  friend bool operator == (const log_double &x,
			   const log_double &y);
  
  friend bool operator != (const log_double &x,
			   const log_double &y);
  
  friend bool operator >= (const log_double &x,
			   const log_double &y);
  
  friend bool operator <= (const log_double &x,
			   const log_double &y);
  
  friend bool operator > (const log_double &x,
			  const log_double &y);
  
  friend bool operator < (const log_double &x,
			  const log_double &y);
  
  /*
  // conversion
  operator double() const {
  return raw_value;
  }

  operator float() const {
    return (float)raw_value;
  }

  operator log_float() const {
    return log_float((float)raw_value);
  }
*/
  // assignment
  log_double &operator = (double &value) {
    raw_value = value;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_double &operator = (float &value) {
    raw_value = (double)value;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }
  log_double &operator = (log_float &value) {
    raw_value = (float)value;
    april_assert(raw_value >= rawscorezero);
    return *this;
  }

};

//---------

// arithmetic operations
log_float operator + (const log_float &x, const log_float &y);
log_float operator - (const log_float &x, const log_float &y);
log_float operator * (const log_float &x, const log_float &y);
log_float operator / (const log_float &x, const log_float &y);

// comparison operations
bool operator == (const log_float &x, const log_float &y);
bool operator != (const log_float &x, const log_float &y);
bool operator >= (const log_float &x, const log_float &y);
bool operator <= (const log_float &x, const log_float &y);
bool operator >  (const log_float &x, const log_float &y);
bool operator <  (const log_float &x, const log_float &y);

// arithmetic operations
log_double operator + (const log_double &x, const log_double &y);
log_double operator - (const log_double &x, const log_double &y);
log_double operator * (const log_double &x, const log_double &y);
log_double operator / (const log_double &x, const log_double &y);

// comparison operations
bool operator == (const log_double &x, const log_double &y);
bool operator != (const log_double &x, const log_double &y);
bool operator >= (const log_double &x, const log_double &y);
bool operator <= (const log_double &x, const log_double &y);
bool operator >  (const log_double &x, const log_double &y);
bool operator <  (const log_double &x, const log_double &y);

//---------------------------------------------------------------

inline log_float& log_float::operator = (const log_double &value) {
  raw_value = (float)value.raw_value;
  april_assert(raw_value >= rawscorezero);
  return *this;
}

// arithmetic operations

inline float logadd1(float x) {
  // invocamos esta función siempre para x<=0
  return log1p(exp(x));
}
inline float log1sub(float x) {
  return log1p(-exp(x));
}

inline log_float operator + (const log_float &x, const log_float &y) {
  log_float result = (x.raw_value > y.raw_value) ?
    log_float(x.raw_value + logadd1(y.raw_value - x.raw_value)) :
    log_float(y.raw_value + logadd1(x.raw_value - y.raw_value));
  assert (result >= log_float::zero());
  return result;
}
inline log_float operator - (const log_float &x, const log_float &y) {
  assert (x.raw_value >= y.raw_value);
  
  log_float result = log_float(x.raw_value + 
			       log1sub(y.raw_value - x.raw_value));
  assert (result >= log_float::zero());
  return result;
}
inline log_float operator * (const log_float &x, const log_float &y) {
  log_float result = log_float(x.raw_value + y.raw_value);
  assert (result >= log_float::zero());
  return result;
}
inline log_float operator / (const log_float &x, const log_float &y) {
  log_float result = log_float(x.raw_value - y.raw_value);
  assert (result >= log_float::zero());
  return result;
}

// comparison operations
inline bool operator == (const log_float &x, const log_float &y) {
  return (x.raw_value == y.raw_value);
}
inline bool operator != (const log_float &x, const log_float &y) {
  return (x.raw_value != y.raw_value);
}
inline bool operator <= (const log_float &x, const log_float &y) {
  return (x.raw_value <= y.raw_value);
}
inline bool operator >= (const log_float &x, const log_float &y) {
  return (x.raw_value >= y.raw_value);
}
inline bool operator <  (const log_float &x, const log_float &y) {
  return (x.raw_value <  y.raw_value);
}
inline bool operator >  (const log_float &x, const log_float &y) {
  return (x.raw_value >  y.raw_value);
}

inline double logadd1(double x) {
  return log1p(exp(x));
}
inline double log1sub(double x) {
  return log1p(-exp(x));
}

inline log_double operator + (const log_double &x, const log_double &y) {
  log_double result = (x.raw_value > y.raw_value) ?
    log_double(x.raw_value + logadd1(y.raw_value - x.raw_value)) :
    log_double(y.raw_value + logadd1(x.raw_value - y.raw_value));
  april_assert(result >= log_float::zero());
  return result;
}
inline log_double operator - (const log_double &x, const log_double &y) {
  if (x.raw_value < y.raw_value) {
    // TODO: report error
  }
  log_double result = log_double(x.raw_value + 
				 log1sub(y.raw_value - x.raw_value));
  april_assert(result >= log_float::zero());
  return result;
}
inline log_double operator * (const log_double &x, const log_double &y) {
  log_double result = log_double(x.raw_value + y.raw_value);
  april_assert(result >= log_float::zero());
  return result;
}
inline log_double operator / (const log_double &x, const log_double &y) {
  log_double result = log_double(x.raw_value - y.raw_value);
  april_assert(result >= log_float::zero());
  return result;
}

// comparison operations
inline bool operator == (const log_double &x, const log_double &y) {
  return (x.raw_value == y.raw_value);
}
inline bool operator != (const log_double &x, const log_double &y) {
  return (x.raw_value != y.raw_value);
}
inline bool operator <= (const log_double &x, const log_double &y) {
  return (x.raw_value <= y.raw_value);
}
inline bool operator >= (const log_double &x, const log_double &y) {
  return (x.raw_value >= y.raw_value);
}
inline bool operator <  (const log_double &x, const log_double &y) {
  return (x.raw_value <  y.raw_value);
}
inline bool operator >  (const log_double &x, const log_double &y) {
  return (x.raw_value >  y.raw_value);
}

// special cases:
inline log_float log_float::operator += (const log_double &y) {
  *this = log_double(*this) + y;
  april_assert(raw_value >= rawscorezero);
  return *this;
}
inline log_float log_float::operator *= (const log_double &y) {
  *this = log_double(*this) * y;
  april_assert(raw_value >= rawscorezero);
  return *this;
}
inline log_float log_float::operator /= (const log_double &y) {
  *this = log_double(*this) / y;
  april_assert(raw_value >= rawscorezero);
  return *this;
}
inline log_float log_float::operator -= (const log_double &y) {
  *this = log_double(*this) - y;
  april_assert(raw_value >= rawscorezero);
  return *this;
}



#endif // LOGBASE_ASSERT_H

