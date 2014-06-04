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
#include "normal_distribution.h"
#include "unused_variable.h"
#include "utilMatrixFloat.h"

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
    if (cov_det.log() < 0.0f)
      cov_det = cov_det.raise_to(-1.0f);
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
      diff_row->axpy(-1.0, mean);
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

  StatisticalDistributionBase *GeneralNormalDistribution::clone() {
    return new GeneralNormalDistribution(mean->clone(), cov->clone());
  }
  
  MatrixFloatSet *GeneralNormalDistribution::getParams() {
    MatrixFloatSet *dict = new MatrixFloatSet();
    dict->insert("mu", mean);
    dict->insert("sigma", cov);
    return dict;
  }
  
  char *GeneralNormalDistribution::toLuaString(bool is_ascii) const {
    buffer_list buffer;
    char *mean_str, *cov_str;
    int len;
    mean_str = writeMatrixFloatToString(mean, is_ascii, len);
    cov_str = writeMatrixFloatToString(cov, is_ascii, len);
    buffer.printf("stats.dist.normal(matrix.fromString[[%s]], matrix.fromString[[%s]])",
                  mean_str, cov_str);
    delete[] mean_str;
    delete[] cov_str;
    return buffer.to_string(buffer_list::NULL_TERMINATED);
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
    for (SparseMatrixFloat::iterator it(inv_cov->begin());
         it != inv_cov->end(); ++it) {
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
    if (cov_det.log() < 0.0f)
      cov_det = cov_det.raise_to(-1.0f);
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

  StatisticalDistributionBase *DiagonalNormalDistribution::clone() {
    return new DiagonalNormalDistribution(mean->clone(), cov->clone());
  }
  
  MatrixFloatSet *DiagonalNormalDistribution::getParams() {
    MatrixFloatSet *dict = new MatrixFloatSet();
    dict->insert("mu", mean);
    dict->insert("sigma", cov);
    return dict;
  }
  
  char *DiagonalNormalDistribution::toLuaString(bool is_ascii) const {
    buffer_list buffer;
    char *mean_str, *cov_str;
    int len;
    mean_str = writeMatrixFloatToString(mean, is_ascii, len);
    cov_str = writeSparseMatrixFloatToString(cov, is_ascii, len);
    buffer.printf("stats.dist.normal(matrix.fromString[[%s]], matrix.sparse.fromString[[%s]])",
                  mean_str, cov_str);
    delete[] mean_str;
    delete[] cov_str;
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
