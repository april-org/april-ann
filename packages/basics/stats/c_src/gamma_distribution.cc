/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
// #include <ctgmath> // FIXME: this library needs c++11 extensions
#include "buffer_list.h"
#include "error_print.h"
#include "gamma_distribution.h"
#include "gamma_variate.h"
#include "matrix_ext_blas.h"
#include "matrix_ext_initializers.h"
#include "matrix_ext_operations.h"
#include "utilMatrixFloat.h"

using AprilUtils::buffer_list;
using AprilUtils::log_float;
using AprilUtils::SharedPtr;
using AprilUtils::UniquePtr;
using Basics::MatrixFloat;
using Basics::MTRand;

using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Operations;

namespace Stats {
  
  GammaDistribution::GammaDistribution(MatrixFloat *alpha, MatrixFloat *beta) :
    StatisticalDistributionBase(1),
    alpha(alpha), beta(beta) {
    if (alpha->getNumDim() != 1 || beta->getNumDim() != 1 ||
        alpha->getDimSize(0) != 1 || beta->getDimSize(0) != 1)
      ERROR_EXIT(128, "Expected alpha,beta one-dimensional matrices with size 1\n");
    updateParams();
  }

  GammaDistribution::~GammaDistribution() {
  }
  
  void GammaDistribution::updateParams() {
    alphaf = (*alpha)(0);
    betaf  = (*beta)(0);
    if (!(alphaf > 0.0f) || !(betaf > 0.0f)) {
      ERROR_EXIT(128, "Gamma distribution needs > 0 alpha and beta params\n");
    }
    // log( beta^alpha / gamma(alpha) ) = alpha * log beta - lgamma(alpha)
    log_float beta_pow_alpha(logf(betaf)*alphaf);
    log_float gamma_a(lgamma(alphaf));
    CTE = beta_pow_alpha / gamma_a;
  }
  
  void GammaDistribution::privateSample(MTRand *rng,
                                        MatrixFloat *result) {
    double scale = 1.0/static_cast<double>(betaf);
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end(); ++result_it) {
      *result_it = gammaVariate(rng, 0.0, scale, static_cast<double>(alphaf));
    }
  }
  
  void GammaDistribution::privateLogpdf(const MatrixFloat *x,
                                        MatrixFloat *result) {
    // pdf = CTE x^(alpha - 1) exp(-beta * x)
    // logpdf = logCTE + (alpha-1)*log(x) - beta*x
    UniquePtr<MatrixFloat> aux_r(result->rightInflate(1));
    matLog(x, aux_r.get());
    matScal(aux_r.get(), alphaf - 1.0f);
    matAxpy(aux_r.get(), -betaf, x);
    matScalarAdd(aux_r.get(), CTE.log());
  }

  void GammaDistribution::privateLogcdf(const MatrixFloat *x,
                                        MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
  }

  StatisticalDistributionBase *GammaDistribution::clone() {
    return new GammaDistribution(alpha->clone(), beta->clone());
  }
  
  const char *GammaDistribution::luaCtorName() const {
    return "stats.dist.gamma";
  }
  int GammaDistribution::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable::pushInto(L, alphaf);
    AprilUtils::LuaTable::pushInto(L, gammaf);
    return 2;
  }
  
}
