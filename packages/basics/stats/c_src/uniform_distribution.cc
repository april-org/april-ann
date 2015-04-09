/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#include "matrixFloat.h"
#include "smart_ptr.h"
#include "uniform_distribution.h"

using AprilUtils::log_float;
using AprilUtils::SharedPtr;
using AprilIO::CStringStream;
using Basics::MatrixFloat;
using Basics::MTRand;

namespace Stats {

  UniformDistribution::UniformDistribution(MatrixFloat *low,
                                           MatrixFloat *high) :
    StatisticalDistributionBase(low->size()),
    low(low), high(high), diff(0) {
    IncRef(low);
    IncRef(high);
    if (!low->sameDim(high))
      ERROR_EXIT(128, "Expected same sizes in low and high matrices\n");
    MatrixFloat::const_iterator low_it(low->begin());
    MatrixFloat::const_iterator high_it(high->begin());
    while(low_it != low->end()) {
      if ( ! (*low_it < *high_it) )
        ERROR_EXIT(128, "All low values must be less than high matrix\n");
      ++low_it;
      ++high_it;
    }
    updateParams();
  }

  UniformDistribution::~UniformDistribution() {
    DecRef(low);
    DecRef(high);
    DecRef(diff);
  }

  void UniformDistribution::privateSample(MTRand *rng, MatrixFloat *result) {
    // traverse in row_major result matrix
    MatrixFloat::iterator it(result->begin());
    while(it != result->end()) {
      // traverse in row_major low and diff matrices
      MatrixFloat::const_iterator low_it(low->begin()), diff_it(diff->begin());
      while(low_it != low->end()) {
        *it = rng->rand(*diff_it) + *low_it;
        ++it;
        ++low_it;
        ++diff_it;
      }
    }
  }

  void UniformDistribution::privateLogpdf(const MatrixFloat *x,
                                          MatrixFloat *result) {
    MatrixFloat::iterator result_it(result->begin());
    MatrixFloat::const_iterator x_it(x->begin());
    while(x_it != x->end()) {
      log_float one = log_float::one();
      log_float current_result = log_float::one();
      MatrixFloat::const_iterator low_it(low->begin()), high_it(high->begin());
      MatrixFloat::const_iterator diff_it(diff->begin());
      while(low_it != low->end() && current_result > log_float::zero()) {
        if (*low_it <= *x_it && *x_it <= *high_it)
          current_result *= one / log_float::from_float(*diff_it);
        else current_result *= log_float::zero();
        ++low_it;
        ++high_it;
        ++x_it;
        ++diff_it;
      }
      while(low_it != low->end()) {
        ++low_it;
        ++high_it;
        ++x_it;
        ++diff_it;
      }
      *result_it = current_result.log();
      ++result_it;
    }
  }

  void UniformDistribution::privateLogcdf(const MatrixFloat *x,
                                          MatrixFloat *result) {
    MatrixFloat::iterator result_it(result->begin());
    MatrixFloat::const_iterator x_it(x->begin());
    while(x_it != x->end()) {
      log_float one = log_float::one();
      log_float current_result = log_float::one();
      MatrixFloat::const_iterator low_it(low->begin()), high_it(high->begin());
      MatrixFloat::const_iterator diff_it(diff->begin());
      while(low_it != low->end() && current_result > log_float::zero()) {
        if (*x_it < *low_it) {
          current_result *= log_float::zero();
        }
        else if (*low_it <= *x_it && *x_it < *high_it) {
          current_result *= log_float::from_float(*x_it - *low_it) / log_float::from_float(*diff_it);
        }
        else {
          current_result *= log_float::one();
        }
        ++low_it;
        ++high_it;
        ++x_it;
        ++diff_it;
      }
      while(low_it != low->end()) {
        ++low_it;
        ++high_it;
        ++x_it;
        ++diff_it;
      }
      *result_it = current_result.log();
      ++result_it;
    }
  }

  StatisticalDistributionBase *UniformDistribution::clone() {
    return new UniformDistribution(low->clone(), high->clone());
  }

  char *UniformDistribution::toLuaString(bool is_ascii) const {
    SharedPtr<CStringStream> stream(new CStringStream());
    AprilUtils::LuaTable options;
    options.put("ascii", is_ascii);
    stream->put("stats.dist.uniform(matrix.fromString[[");
    low->write(stream.get(), options);
    stream->put("]], matrix.fromString[[");
    high->write(stream.get(), options);
    stream->put("]])\0",4);
    return stream->releaseString();
  }
  
  void UniformDistribution::updateParams() {
    AssignRef(diff, high->clone());
    AprilMath::MatrixExt::BLAS::matAxpy(diff, -1.0f, low);
  }
}
