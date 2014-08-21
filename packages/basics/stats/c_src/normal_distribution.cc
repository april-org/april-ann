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
#define _USE_MATH_DEFINES
#include <cmath>
#include "buffer_list.h"
#include "check_floats.h"
#include "error_print.h"
#include "matrixFloat.h"
#include "normal_distribution.h"
#include "smart_ptr.h"
#include "sparse_matrixFloat.h"
#include "unused_variable.h"

using april_utils::log_float;
using april_utils::SharedPtr;
using AprilIO::CStringStream;
using basics::MatrixFloat;
using basics::SparseMatrixFloat;
using basics::MTRand;

namespace Stats {
  
  const float M_2PI = M_PI*2.0f;
  
  GeneralNormalDistribution::GeneralNormalDistribution(MatrixFloat *mean,
                                                       MatrixFloat *cov) :
    StatisticalDistributionBase(mean->size()),
    mean(mean), cov(cov), inv_cov(0), L(0) {
    if (mean->getNumDim() != 1)
      ERROR_EXIT(128, "Expected one-dimensional mean matrix\n");
    if (cov->getNumDim() != 2 || cov->getDimSize(0) != cov->getDimSize(1))
      ERROR_EXIT(128, "Expected squared bi-dimensional cov matrix\n");
    if (mean->getDimSize(0) != cov->getDimSize(0))
      ERROR_EXIT(128, "Expected mean and cov matrix with same size\n");
    if (mean->getMajorOrder() != CblasColMajor ||
        cov->getMajorOrder() != CblasColMajor)
      ERROR_EXIT(128, "Expected col_major matrices\n");
    IncRef(mean);
    IncRef(cov);
    updateParams();
  }

  GeneralNormalDistribution::~GeneralNormalDistribution() {
    DecRef(mean);
    DecRef(cov);
    DecRef(inv_cov);
    if (L != 0) DecRef(L);
  }
  
  void GeneralNormalDistribution::updateParams() {
    // K = 1 / sqrtf( 2*pi^k * |cov| )
    AssignRef(inv_cov, cov->inv());
    // TODO: check covariance matrix to be definite positive
    cov_det = cov->logDeterminant(cov_det_sign);
    log_float KM_2PI = log_float::from_float(M_2PI).
      raise_to(static_cast<float>(mean->getDimSize(0)));
    log_float denom = (KM_2PI * cov_det).raise_to(0.5f);
    K = log_float::one() / denom;
    if (L) DecRef(L);
    L = 0;
  }
  
  void GeneralNormalDistribution::privateSample(MTRand *rng,
                                                MatrixFloat *result) {
    if (L == 0) {
      L = cov->cholesky('L');
      IncRef(L);
    }
    MatrixFloat *z = result->cloneOnlyDims();
    IncRef(z);
    for (MatrixFloat::iterator z_it = z->begin(); z_it != z->end(); ++z_it) {
      *z_it = static_cast<float>(rng->randNorm(0.0, 1.0));
    }
    result->gemm(CblasNoTrans, CblasNoTrans, 1.0f, z, L, 0.0f);
    DecRef(z);
    MatrixFloat *result_row = 0;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row);
      result_row->axpy(1.0, mean);
    }
    delete result_row;
  }
  
  void GeneralNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                MatrixFloat *result) {
    MatrixFloat *diff = x->clone();
    IncRef(diff);
    int dims[1] = { x->getDimSize(1) };
    MatrixFloat *mult = new MatrixFloat(1, dims, CblasColMajor);
    IncRef(mult);
    // over all samples (bunch_size)
    MatrixFloat::iterator result_it(result->begin());
    MatrixFloat *diff_row = 0;
    for (int i=0; i<x->getDimSize(0); ++i, ++result_it) {
      diff_row = diff->select(0, i, diff_row);
      diff_row->axpy(-1.0f, mean);
      mult->gemv(CblasNoTrans, 1.0f, inv_cov, diff_row, 0.0f);
      *result_it = -0.5 * mult->dot(diff_row);
    }
    result->scalarAdd( K.log() );
    delete diff_row;
    DecRef(diff);
    DecRef(mult);
  }

  void GeneralNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  void GeneralNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                          MatrixFloat *result) {
    MatrixFloat *diff = x->clone();
    IncRef(diff);
    // over all samples (bunch_size)
    MatrixFloat *diff_row = 0;
    MatrixFloat *result_row = 0;
    for (int i=0; i<x->getDimSize(0); ++i) {
      diff_row = diff->select(0, i, diff_row);
      result_row = result->select(0, i, result_row);
      diff_row->axpy(-1.0, mean);
      result_row->gemv(CblasNoTrans, -1.0f, inv_cov, diff_row, 1.0f);
    }
    delete diff_row;
    delete result_row;
    DecRef(diff);
  }

  StatisticalDistributionBase *GeneralNormalDistribution::clone() {
    return new GeneralNormalDistribution(mean->clone(), cov->clone());
  }
  
  char *GeneralNormalDistribution::toLuaString(bool is_ascii) const {
    SharedPtr<CStringStream> stream(new CStringStream());
    april_utils::HashTableOptions options;
    options.putBoolean("ascii", is_ascii);
    stream->put("stats.dist.normal(matrix.fromString[[");
    mean->write(stream.get(), &options);
    stream->put("]], matrix.fromString[[");
    cov->write(stream.get(), &options);
    stream->put("]])\0",4);
    return stream->releaseString();
  }

  ////////////////////////////////////////////////////////////////////////////

  DiagonalNormalDistribution::DiagonalNormalDistribution(MatrixFloat *mean,
                                                         SparseMatrixFloat *cov) :
    StatisticalDistributionBase(mean->size()),
    mean(mean), cov(cov), inv_cov(0), L(0) {
    if (mean->getNumDim() != 1)
      ERROR_EXIT(128, "Expected one-dimensional mean matrix\n");
    if (cov->getNumDim() != 2 || cov->getDimSize(0) != cov->getDimSize(1))
      ERROR_EXIT(128, "Expected squared bi-dimensional cov matrix\n");
    if (mean->getDimSize(0) != cov->getDimSize(0))
      ERROR_EXIT(128, "Expected mean and cov matrix with same size\n");
    if (mean->getMajorOrder() != CblasColMajor)
      ERROR_EXIT(128, "Expected col_major mean matrix\n");
    IncRef(mean);
    IncRef(cov);
    updateParams();
  }

  DiagonalNormalDistribution::~DiagonalNormalDistribution() {
    DecRef(mean);
    DecRef(cov);
    DecRef(inv_cov);
    if (L != 0) DecRef(L);
  }
  
  void DiagonalNormalDistribution::updateParams() {
    if (!cov->isDiagonal())
      ERROR_EXIT(256, "Expected diagonal cov sparse matrix\n");
    // K = 1 / sqrtf( 2*pi^k * |cov| )
    AssignRef(inv_cov, cov->clone());
    inv_cov->div(1.0f);
    // inv_cov->pruneSubnormalAndCheckNormal();
    cov_det = log_float::one();
    for (SparseMatrixFloat::iterator it(cov->begin());
         it != cov->end(); ++it) {
      if (!std::isfinite(*it)) {
        int x0,x1;
        it.getCoords(x0,x1);
	ERROR_EXIT3(256, "No finite number at position %d,%d with value %g\n",
                    x0, x1, *it);
      }
      if (*it < 0.0f)
        ERROR_EXIT(256, "Expected a definite positive covariance matrix\n");
      cov_det *= log_float::from_float(*it);
    }
    log_float KM_2PI = log_float::from_float(M_2PI).
      raise_to(static_cast<float>(mean->getDimSize(0)));
    log_float denom = (KM_2PI * cov_det).raise_to(0.5f);
    K = log_float::one() / denom;
    if (L != 0) DecRef(L);
    L = 0;
  }
  
  void DiagonalNormalDistribution::privateSample(MTRand *rng,
                                                 MatrixFloat *result) {
    if (L == 0) {
      L = cov->clone();
      L->sqrt();
      IncRef(L);
    }
    MatrixFloat *z = result->cloneOnlyDims();
    IncRef(z);
    for (MatrixFloat::iterator z_it = z->begin(); z_it != z->end(); ++z_it) {
      *z_it = static_cast<float>(rng->randNorm(0.0, 1.0));
    }
    result->sparseMM(CblasTrans, CblasTrans, CblasTrans, 1.0f, L, z, 0.0f);
    DecRef(z);
    MatrixFloat *result_row = 0;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row);
      result_row->axpy(1.0, mean);
    }
    delete result_row;
  }
  
  void DiagonalNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                 MatrixFloat *result) {
    MatrixFloat *diff = x->clone();
    IncRef(diff);
    int dims[1] = { x->getDimSize(1) };
    MatrixFloat *mult = new MatrixFloat(1, dims, CblasColMajor);
    IncRef(mult);
    // over all samples (bunch_size)
    MatrixFloat::iterator result_it(result->begin());
    MatrixFloat *diff_row = 0;
    for (int i=0; i<x->getDimSize(0); ++i, ++result_it) {
      diff_row = diff->select(0, i, diff_row);
      diff_row->axpy(-1.0, mean);
      mult->gemv(CblasNoTrans, 1.0f, inv_cov, diff_row, 0.0f);
      *result_it = -0.5 * mult->dot(diff_row);
    }
    result->scalarAdd( K.log() );
    delete diff_row;
    DecRef(diff);
    DecRef(mult);
  }

  void DiagonalNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                 MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  void DiagonalNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                           MatrixFloat *result) {
    MatrixFloat *diff = x->clone();
    IncRef(diff);
    // over all samples (bunch_size)
    MatrixFloat *diff_row = 0;
    MatrixFloat *result_row = 0;
    for (int i=0; i<x->getDimSize(0); ++i) {
      diff_row = diff->select(0, i, diff_row);
      result_row = result->select(0, i, result_row);
      diff_row->axpy(-1.0, mean);
      result_row->gemv(CblasNoTrans, -1.0f, inv_cov, diff_row, 1.0f);
    }
    delete diff_row;
    delete result_row;
    DecRef(diff);
  }

  StatisticalDistributionBase *DiagonalNormalDistribution::clone() {
    return new DiagonalNormalDistribution(mean->clone(), cov->clone());
  }
  
  char *DiagonalNormalDistribution::toLuaString(bool is_ascii) const {
    SharedPtr<CStringStream> stream(new CStringStream());
    april_utils::HashTableOptions options;
    options.putBoolean("ascii", is_ascii);
    stream->put("stats.dist.normal(matrix.fromString[[");
    mean->write(stream.get(), &options);
    stream->put("]], matrix.sparse.fromString[[");
    cov->write(stream.get(), &options);
    stream->put("]])\0",4); // forces a \0 at the end of the buffer
    return stream->releaseString();
  }
  
  ////////////////////////////////////////////////////////////////////////////

  StandardNormalDistribution::StandardNormalDistribution() :
    StatisticalDistributionBase(1) {
    K = log_float::from_float(1.0f/sqrtf(M_2PI));
  }

  StandardNormalDistribution::~StandardNormalDistribution() {
  }
  
  void StandardNormalDistribution::privateSample(MTRand *rng,
                                                 MatrixFloat *result) {
    for (MatrixFloat::iterator result_it = result->begin();
         result_it != result->end(); ++result_it) {
      *result_it = static_cast<float>(rng->randNorm(0.0, 1.0));
    }
  }
  
  void StandardNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                MatrixFloat *result) {
    MatrixFloat *x2 = x->clone();
    IncRef(x2);
    x2->pow(2);
    result->fill(K.log());
    result->axpy(-0.5,  x2);
    DecRef(x2);
  }

  void StandardNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  void StandardNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                           MatrixFloat *result) {
    result->axpy(-1.0, x);
  }

  StatisticalDistributionBase *StandardNormalDistribution::clone() {
    return new StandardNormalDistribution();
  }
  
  char *StandardNormalDistribution::toLuaString(bool is_ascii) const {
    UNUSED_VARIABLE(is_ascii);
    april_utils::buffer_list buffer;
    buffer.printf("stats.dist.normal()");
    return buffer.to_string(april_utils::buffer_list::NULL_TERMINATED);
  }

  ////////////////////////////////////////////////////////////////////////////  

  GeneralLogNormalDistribution::GeneralLogNormalDistribution(MatrixFloat *mean,
                                                             MatrixFloat *cov,
                                                             MatrixFloat *location) :
    GeneralNormalDistribution(mean,cov),
    location(location) {
    if (location == 0) {
      location = mean->cloneOnlyDims();
      location->zeros();
      this->location = location;
    }
    IncRef(location);
    if (!location->sameDim(mean))
      ERROR_EXIT(256, "Expected location param with same shape as mean param\n");
    updateParams();
  }

  GeneralLogNormalDistribution::~GeneralLogNormalDistribution() {
    DecRef(location);
  }
  
  void GeneralLogNormalDistribution::privateSample(MTRand *rng,
                                                   MatrixFloat *result) {
    GeneralNormalDistribution::privateSample(rng, result);
    result->exp();
    MatrixFloat *result_row = 0;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row);
      result_row->axpy(1.0, location);
    }
    delete result_row;
  }
  
  void GeneralLogNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                   MatrixFloat *result) {
    MatrixFloat *xlog = x->clone();
    IncRef(xlog);
    MatrixFloat *xlog_row = 0;
    for (int i=0; i<x->getDimSize(0); ++i) {
      xlog_row = xlog->select(0, i, xlog_row);
      xlog_row->axpy(-1.0f, location);
    }
    delete xlog_row;
    xlog->log();
    GeneralNormalDistribution::privateLogpdf(xlog, result);
    MatrixFloat *xlog_sum = xlog->sum(1);
    IncRef(xlog_sum);
    result->axpy(-1.0f, xlog_sum);
    DecRef(xlog);
    DecRef(xlog_sum);
  }

  void GeneralLogNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                   MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  void GeneralLogNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                             MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  StatisticalDistributionBase *GeneralLogNormalDistribution::clone() {
    return new GeneralLogNormalDistribution(mean->clone(), cov->clone(),
                                            location->clone());
  }
  
  char *GeneralLogNormalDistribution::toLuaString(bool is_ascii) const {
    SharedPtr<CStringStream> stream(new CStringStream());
    april_utils::HashTableOptions options;
    options.putBoolean("ascii", is_ascii);
    stream->put("stats.dist.lognormal(matrix.fromString[[");
    mean->write(stream.get(), &options);
    stream->put("]], matrix.fromString[[");
    cov->write(stream.get(), &options);
    stream->put("]], matrix.fromString[[");
    location->write(stream.get(), &options);
    stream->put("]])\0",4); // forces a \0 at the end of the buffer
    return stream->releaseString();
  }

  ////////////////////////////////////////////////////////////////////////////

  DiagonalLogNormalDistribution::DiagonalLogNormalDistribution(MatrixFloat *mean,
                                                               SparseMatrixFloat *cov,
                                                               MatrixFloat *location) :
    DiagonalNormalDistribution(mean, cov),
    location(location) {
    if (location == 0) {
      location = mean->cloneOnlyDims();
      location->zeros();
      this->location = location;
    }
    IncRef(location);
    if (!location->sameDim(mean))
      ERROR_EXIT(256, "Expected location param with same shape as mean param\n");
    updateParams();
  }

  DiagonalLogNormalDistribution::~DiagonalLogNormalDistribution() {
    DecRef(location);
  }
  
  void DiagonalLogNormalDistribution::privateSample(MTRand *rng,
                                                    MatrixFloat *result) {
    DiagonalNormalDistribution::privateSample(rng, result);
    result->exp();
    MatrixFloat *result_row = 0;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row);
      result_row->axpy(1.0, location);
    }
    delete result_row;
  }
  
  void DiagonalLogNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                    MatrixFloat *result) {
    MatrixFloat *xlog = x->clone();
    IncRef(xlog);
    MatrixFloat *xlog_row = 0;
    for (int i=0; i<x->getDimSize(0); ++i) {
      xlog_row = xlog->select(0, i, xlog_row);
      xlog_row->axpy(-1.0f, location);
    }
    delete xlog_row;
    xlog->log();
    DiagonalNormalDistribution::privateLogpdf(xlog, result);
    MatrixFloat *xlog_sum = xlog->sum(1);
    IncRef(xlog_sum);
    result->axpy(-1.0f, xlog_sum);
    DecRef(xlog);
    DecRef(xlog_sum);
  }

  void DiagonalLogNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                    MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  void DiagonalLogNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                              MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  StatisticalDistributionBase *DiagonalLogNormalDistribution::clone() {
    return new DiagonalLogNormalDistribution(mean->clone(), cov->clone(),
                                             location->clone());
  }
  
  char *DiagonalLogNormalDistribution::toLuaString(bool is_ascii) const {
    SharedPtr<CStringStream> stream(new CStringStream());
    april_utils::HashTableOptions options;
    options.putBoolean("ascii", is_ascii);
    stream->put("stats.dist.lognormal(matrix.fromString[[");
    mean->write(stream.get(), &options);
    stream->put("]], matrix.sparse.fromString[[");
    cov->write(stream.get(), &options);
    stream->put("]], matrix.fromString[[");
    location->write(stream.get(), &options);
    stream->put("]])\0",4); // forces a \0 at the end of the buffer
    return stream->releaseString();
  }
  
}
