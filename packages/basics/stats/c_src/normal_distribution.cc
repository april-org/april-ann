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

using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::LAPACK;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using AprilUtils::log_float;
using AprilUtils::SharedPtr;
using AprilIO::CStringStream;
using Basics::MatrixFloat;
using Basics::SparseMatrixFloat;
using Basics::MTRand;

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
    AssignRef(inv_cov, matInv(cov));
    // TODO: check covariance matrix to be definite positive
    cov_det = matLogDeterminant(cov, cov_det_sign);
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
      L = matCholesky(cov, 'L');
      IncRef(L);
    }
    MatrixFloat *z = result->cloneOnlyDims();
    IncRef(z);
    for (MatrixFloat::iterator z_it = z->begin(); z_it != z->end(); ++z_it) {
      *z_it = static_cast<float>(rng->randNorm(0.0, 1.0));
    }
    matGemm(result, CblasNoTrans, CblasNoTrans, 1.0f, z, L, 0.0f);
    DecRef(z);
    MatrixFloat *result_row = 0;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row);
      matAxpy(result_row, 1.0f, mean);
    }
    delete result_row;
  }
  
  void GeneralNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                MatrixFloat *result) {
    SharedPtr<MatrixFloat> diff( x->clone() );
    int dims[1] = { x->getDimSize(1) };
    SharedPtr<MatrixFloat> mult( new MatrixFloat(1, dims) );
    // over all samples (bunch_size)
    MatrixFloat::iterator result_it(result->begin());
    SharedPtr<MatrixFloat> diff_row;
    for (int i=0; i<x->getDimSize(0); ++i, ++result_it) {
      diff_row = diff->select(0, i, diff_row.get());
      matAxpy(diff_row.get(), -1.0f, mean);
      matGemv(mult.get(), CblasNoTrans, 1.0f, inv_cov, diff_row.get(), 0.0f);
      *result_it = -0.5 * matDot(mult.get(), diff_row.get());
    }
    matScalarAdd(result, K.log() );
  }

  void GeneralNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "Not implemented");
  }

  void GeneralNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                          MatrixFloat *result) {
    SharedPtr<MatrixFloat> diff( x->clone() );
    // over all samples (bunch_size)
    SharedPtr<MatrixFloat> diff_row;
    SharedPtr<MatrixFloat> result_row;
    for (int i=0; i<x->getDimSize(0); ++i) {
      diff_row = diff->select(0, i, diff_row.get());
      result_row = result->select(0, i, result_row.get());
      matAxpy(diff_row.get(), -1.0f, mean);
      matGemv(result_row.get(), CblasNoTrans,
              -1.0f, inv_cov, diff_row.get(), 1.0f);
    }
  }

  StatisticalDistributionBase *GeneralNormalDistribution::clone() {
    return new GeneralNormalDistribution(mean->clone(), cov->clone());
  }

  const char *GeneralNormalDistribution::luaCtorName() const {
    return "stats.dist.normal";
  }
  int GeneralNormalDistribution::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable::pushInto(L, mean);
    AprilUtils::LuaTable::pushInto(L, cov);
    return 2;
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
    matDiv(inv_cov, 1.0f);
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
      matSqrt(L);
      IncRef(L);
    }
    SharedPtr<MatrixFloat> z( result->cloneOnlyDims() );
    for (MatrixFloat::iterator z_it = z->begin(); z_it != z->end(); ++z_it) {
      *z_it = static_cast<float>(rng->randNorm(0.0, 1.0));
    }
    SharedPtr<MatrixFloat> rT(result->transpose());
    matSparseMM(rT.get(), CblasTrans, CblasTrans,
                1.0f, L, z.get(), 0.0f);
    SharedPtr<MatrixFloat> result_row;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row.get());
      matAxpy(result_row.get(), 1.0f, mean);
    }
  }
  
  void DiagonalNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                 MatrixFloat *result) {
    SharedPtr<MatrixFloat> diff( x->clone() );
    int dims[1] = { x->getDimSize(1) };
    SharedPtr<MatrixFloat> mult( new MatrixFloat(1, dims) );
    // over all samples (bunch_size)
    MatrixFloat::iterator result_it(result->begin());
    SharedPtr<MatrixFloat> diff_row;
    for (int i=0; i<x->getDimSize(0); ++i, ++result_it) {
      diff_row = diff->select(0, i, diff_row.get());
      matAxpy(diff_row.get(), -1.0f, mean);
      matGemv(mult.get(), CblasNoTrans, 1.0f, inv_cov, diff_row.get(), 0.0f);
      *result_it = -0.5 * matDot(mult.get(), diff_row.get());
    }
    matScalarAdd(result, K.log() );
  }

  void DiagonalNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                 MatrixFloat *result) {
    if (getSize() != 1) {
      ERROR_EXIT(128, "Only implemented for univariate distributions\n");
    }
    float mu  = *(mean->begin());
    float sd  = ::sqrtf(*(cov->begin()));
    static float inv_sqrt2 = 1.0f/(sd*sqrtf(2.0f));
    MatrixFloat::const_iterator x_it = x->begin();
    for (MatrixFloat::iterator it=result->begin(); it!=result->end(); ++it) {
      *it = AprilMath::m_log( 0.5f * (1.0f + ::erff( (*x_it - mu) * inv_sqrt2 )) );
    }
  }

  void DiagonalNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                           MatrixFloat *result) {
    SharedPtr<MatrixFloat> diff( x->clone() );
    // over all samples (bunch_size)
    SharedPtr<MatrixFloat> diff_row;
    SharedPtr<MatrixFloat> result_row;
    for (int i=0; i<x->getDimSize(0); ++i) {
      diff_row = diff->select(0, i, diff_row.get());
      result_row = result->select(0, i, result_row.get());
      matAxpy(diff_row.get(), -1.0f, mean);
      matGemv(result_row.get(), CblasNoTrans, -1.0f,
              inv_cov, diff_row.get(), 1.0f);
    }
  }

  StatisticalDistributionBase *DiagonalNormalDistribution::clone() {
    return new DiagonalNormalDistribution(mean->clone(), cov->clone());
  }

  const char *DiagonalNormalDistribution::luaCtorName() const {
    return "stats.dist.normal";
  }
  int DiagonalNormalDistribution::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable::pushInto(L, mean);
    AprilUtils::LuaTable::pushInto(L, cov);
    return 2;
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
    SharedPtr<MatrixFloat> x2( x->clone() );
    matPow(x2.get(), 2.0f);
    matFill(result, K.log());
    matAxpy(result, -0.5f,  x2.get());
  }

  void StandardNormalDistribution::privateLogcdf(const MatrixFloat *x,
                                                 MatrixFloat *result) {
    static float inv_sqrt2 = 1.0f/sqrtf(2.0f);
    MatrixFloat::const_iterator x_it = x->begin();
    for (MatrixFloat::iterator it=result->begin(); it!=result->end(); ++it) {
      *it = AprilMath::m_log( 0.5f * (1.0f + ::erff( *x_it * inv_sqrt2 )) );
    }
  }

  void StandardNormalDistribution::privateLogpdfDerivative(const MatrixFloat *x,
                                                           MatrixFloat *result) {
    matAxpy(result, -1.0f, x);
  }

  StatisticalDistributionBase *StandardNormalDistribution::clone() {
    return new StandardNormalDistribution();
  }

  const char *StandardNormalDistribution::luaCtorName() const {
    return "stats.dist.normal";
  }
  int StandardNormalDistribution::exportParamsToLua(lua_State *L) {
    UNUSED_VARIABLE(L);
    return 0;
  }
  
  ////////////////////////////////////////////////////////////////////////////  

  GeneralLogNormalDistribution::GeneralLogNormalDistribution(MatrixFloat *mean,
                                                             MatrixFloat *cov,
                                                             MatrixFloat *location) :
    GeneralNormalDistribution(mean,cov),
    location(location) {
    if (location == 0) {
      location = mean->cloneOnlyDims();
      matZeros(location);
      this->location = location;
    }
    IncRef(location);
    if (!location->sameDim(mean)) {
      ERROR_EXIT(256, "Expected location param with same shape as mean param\n");
    }
    updateParams();
  }

  GeneralLogNormalDistribution::~GeneralLogNormalDistribution() {
    DecRef(location);
  }
  
  void GeneralLogNormalDistribution::privateSample(MTRand *rng,
                                                   MatrixFloat *result) {
    GeneralNormalDistribution::privateSample(rng, result);
    matExp(result);
    SharedPtr<MatrixFloat> result_row;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row.get());
      matAxpy(result_row.get(), 1.0f, location);
    }
  }
  
  void GeneralLogNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                   MatrixFloat *result) {
    SharedPtr<MatrixFloat> xlog( x->clone() );
    SharedPtr<MatrixFloat> xlog_row;
    for (int i=0; i<x->getDimSize(0); ++i) {
      xlog_row = xlog->select(0, i, xlog_row.get());
      matAxpy(xlog_row.get(), -1.0f, location);
    }
    matLog(xlog.get());
    GeneralNormalDistribution::privateLogpdf(matLog(xlog.get()), result);
    SharedPtr<MatrixFloat> xlog_sum( matSum(xlog.get(), 1) );
    matAxpy(result, -1.0f, xlog_sum.get());
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

  const char *GeneralLogNormalDistribution::luaCtorName() const {
    return "stats.dist.lognormal";
  }
  int GeneralLogNormalDistribution::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable::pushInto(L, mean);
    AprilUtils::LuaTable::pushInto(L, cov);
    AprilUtils::LuaTable::pushInto(L, location);
    return 3;
  }
  
  ////////////////////////////////////////////////////////////////////////////

  DiagonalLogNormalDistribution::DiagonalLogNormalDistribution(MatrixFloat *mean,
                                                               SparseMatrixFloat *cov,
                                                               MatrixFloat *location) :
    DiagonalNormalDistribution(mean, cov),
    location(location) {
    if (location == 0) {
      location = mean->cloneOnlyDims();
      matZeros(location);
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
    matExp(result);
    SharedPtr<MatrixFloat> result_row;
    for (int i=0; i<result->getDimSize(0); ++i) {
      result_row = result->select(0, i, result_row.get());
      matAxpy(result_row.get(), 1.0f, location);
    }
  }
  
  void DiagonalLogNormalDistribution::privateLogpdf(const MatrixFloat *x,
                                                    MatrixFloat *result) {
    SharedPtr<MatrixFloat> xlog( x->clone() );
    SharedPtr<MatrixFloat> xlog_row;
    for (int i=0; i<x->getDimSize(0); ++i) {
      xlog_row = xlog->select(0, i, xlog_row.get());
      matAxpy(xlog_row.get(), -1.0f, location);
    }
    matLog(xlog.get());
    DiagonalNormalDistribution::privateLogpdf(xlog.get(), result);
    SharedPtr<MatrixFloat> xlog_sum( matSum(xlog.get(), 1) );
    matAxpy(result, -1.0f, xlog_sum.get());
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

  const char *DiagonalLogNormalDistribution::luaCtorName() const {
    return "stats.dist.lognormal";
  }
  int DiagonalLogNormalDistribution::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable::pushInto(L, mean);
    AprilUtils::LuaTable::pushInto(L, cov);
    AprilUtils::LuaTable::pushInto(L, location);
    return 3;
  }
  
}
