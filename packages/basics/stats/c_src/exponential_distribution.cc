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
#include <cmath>
#include "c_string.h"
#include "error_print.h"
#include "exponential_distribution.h"
#include "matrix_operations.h"
#include "smart_ptr.h"
#include "utilMatrixFloat.h"

using namespace AprilMath::MatrixExt::Operations;
using AprilUtils::log_float;
using AprilUtils::SharedPtr;
using AprilIO::CStringStream;
using Basics::MatrixFloat;
using Basics::MTRand;

namespace Stats {
  
  ExponentialDistribution::ExponentialDistribution(MatrixFloat *lambda) :
    StatisticalDistributionBase(lambda->size()),
    lambda(lambda), inv_lambda(0) {
    if (lambda->getNumDim() != 1)
      ERROR_EXIT(128, "Expected one-dimensional lambda matrix\n");
    if (lambda->getMajorOrder() != CblasColMajor)
      ERROR_EXIT(128, "Expected col_major matrix\n");
    IncRef(lambda);
    updateParams();
  }

  ExponentialDistribution::~ExponentialDistribution() {
    DecRef(lambda);
    if (inv_lambda != 0) DecRef(inv_lambda);
  }
  
  void ExponentialDistribution::updateParams() {
    for (MatrixFloat::iterator it(lambda->begin()); it != lambda->end(); ++it) {
      if (*it < 0.0f) {
        ERROR_EXIT1(128, "Found negative lambda parameter at position %d\n",
                    it.getIdx());
      }
    }
    MatrixFloat *log_lambda = lambda->clone();
    IncRef(log_lambda);
    matLog(log_lambda);
    lambda_prod = log_float(matSum(log_lambda));
    DecRef(log_lambda);
    if (inv_lambda) {
      DecRef(inv_lambda);
      inv_lambda = 0;
    }
  }
  
  void ExponentialDistribution::privateSample(MTRand *rng,
                                              MatrixFloat *result) {
    if (inv_lambda == 0) {
      AssignRef(inv_lambda, lambda->clone());
      matDiv(inv_lambda,1.0f);
    }
    for (MatrixFloat::iterator it(result->begin()); it != result->end(); ++it) {
      float v = static_cast<float>(rng->randDblExc());
      april_assert(v > 0.0f && v < 1.0f);
      *it = v;
    }
    matLog(result);
    matScal(result,-1.0f);
    MatrixFloat *result_row = 0;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row);
      matCmul(result_row, inv_lambda);
    }
    delete result_row;
  }
  
  void ExponentialDistribution::privateLogpdf(const MatrixFloat *x,
                                              MatrixFloat *result) {
    int a,b;
    if (matMin(x,a,b) < 0.0f) {
      ERROR_EXIT(128, "Exponential dist. is not defined for neg. numbers\n");
    }
    UNUSED_VARIABLE(a);
    UNUSED_VARIABLE(b);
    matFill(result, lambda_prod.log());
    matGemv(result, CblasNoTrans, -1.0f, x, lambda, 1.0f);
  }

  void ExponentialDistribution::privateLogcdf(const MatrixFloat *x,
                                              MatrixFloat *result) {
    int a,b;
    if (matMin(x,a,b) < 0.0f)
      ERROR_EXIT(128, "Exponential dist. is not defined for neg. numbers\n");
    UNUSED_VARIABLE(a);
    UNUSED_VARIABLE(b);
    int dims[2] = { result->getDimSize(0), 1 };
    // FIXME: needs a contiguous result matrix
    SharedPtr<MatrixFloat> rewrapped_result( result->rewrap(dims, 2) );
    SharedPtr<MatrixFloat> x_clone( x->clone() ), x_row;
    //
    for (int i=0; i<x->getDimSize(0); ++i) {
      x_row = x_clone->select(0, i, x_row.get());
      matCmul(x_row.get(), lambda);
    }
    matScal(x_clone.get(), -1.0f);
    matExp(x_clone.get());
    matScal(x_clone.get(), -1.0f);
    matLog1p(x_clone.get());
    matSum(x_clone.get(), 1, rewrapped_result.get());
  }

  StatisticalDistributionBase *ExponentialDistribution::clone() {
    return new ExponentialDistribution(lambda->clone());
  }
  
  char *ExponentialDistribution::toLuaString(bool is_ascii) const {
    SharedPtr<CStringStream> stream(new CStringStream());
    stream->put("stats.dist.exponential(matrix.fromString[[");
    AprilUtils::HashTableOptions options;
    lambda->write( stream.get(), options.putBoolean("ascii", is_ascii) );
    stream->put("]])\0",4); // forces a \0 at the end of the buffer
    return stream->releaseString();
  }
  
}
