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
//BIND_HEADER_C
#include "bind_matrix.h"
#include "bind_sparse_matrix.h"
#include "bind_mtrand.h"
//BIND_END

//BIND_HEADER_H
#include "beta_distribution.h"
#include "binomial_distribution.h"
#include "combinations.h"
#include "exponential_distribution.h"
#include "statistical_distribution.h"
#include "uniform_distribution.h"
#include "normal_distribution.h"

using namespace Stats;

//BIND_END

//BIND_FUNCTION stats.comb
{
  unsigned int n, k;
  LUABIND_GET_PARAMETER(1, uint, n);
  LUABIND_GET_PARAMETER(2, uint, k);
  LUABIND_RETURN(uint, Combinations::get(static_cast<unsigned int>(n),
                                         static_cast<unsigned int>(k)));
}
//BIND_END

//BIND_FUNCTION stats.log_comb
{
  unsigned int n, k;
  LUABIND_GET_PARAMETER(1, uint, n);
  LUABIND_GET_PARAMETER(2, uint, k);
  LUABIND_RETURN(double, static_cast<double>
                 (Combinations::lget(static_cast<unsigned int>(n),
                                     static_cast<unsigned int>(k))));
}
//BIND_END

//BIND_FUNCTION stats.lgamma
{
  double x;
  LUABIND_GET_PARAMETER(1, double, x);
  LUABIND_RETURN(double, lgamma(x));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StatisticalDistributionBase stats.dist
//BIND_CPP_CLASS    StatisticalDistributionBase

//BIND_CONSTRUCTOR StatisticalDistributionBase
{
  LUABIND_ERROR("Abstract class");
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase sample
{
  MTRand *rng;
  MatrixFloat *dest;
  LUABIND_GET_PARAMETER(1, MTRand, rng);
  int result_pos = 2, N=-1;
  if (lua_isnumber(L,2)) {
    LUABIND_GET_PARAMETER(2, int, N);
    if (N < 1) LUABIND_ERROR("Expected > 0 number as 2nd argument\n");
    result_pos++;
  }
  LUABIND_GET_OPTIONAL_PARAMETER(result_pos, MatrixFloat, dest, 0);
  if (N > 0) {
    if (dest == 0) {
      int dims[2] = { N, static_cast<int>(obj->getSize()) };
      dest = new MatrixFloat(2, dims);
    }
    else {
      if (N != dest->getDimSize(0))
        LUABIND_FERROR2("Expected a matrix with %d rows, found %d rows\n",
                        N, dest->getDimSize(0));
    }
  }
  LUABIND_RETURN(MatrixFloat, obj->sample(rng, dest));
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase logpdf
{
  MatrixFloat *x, *dest;
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, x);
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
  }
  else {
    float xf;
    LUABIND_CHECK_ARGN(==,1);
    LUABIND_GET_PARAMETER(1, float, xf);
    dest = 0;
    int dims[2] = {1,1};
    x = new MatrixFloat(2, dims);
    (*x)(0,0) = xf;
  }
  IncRef(x);
  LUABIND_RETURN(MatrixFloat, obj->logpdf(x, dest));
  DecRef(x);
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase logcdf
{
  MatrixFloat *x, *dest;
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, x);
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
  }
  else {
    float xf;
    LUABIND_CHECK_ARGN(==,1);
    LUABIND_GET_PARAMETER(1, float, xf);
    dest = 0;
    int dims[2] = {1,1};
    x = new MatrixFloat(2, dims);
    (*x)(0,0) = xf;
  }
  IncRef(x);
  LUABIND_RETURN(MatrixFloat, obj->logcdf(x, dest));
  DecRef(x);
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase logpdf_derivative
{
  MatrixFloat *x, *grads;
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, x);
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, grads, 0);
  }
  else {
    float xf;
    LUABIND_CHECK_ARGN(==,1);
    LUABIND_GET_PARAMETER(1, float, xf);
    grads = 0;
    int dims[2] = {1,1};
    x = new MatrixFloat(2, dims);
    (*x)(0,0) = xf;
  }
  IncRef(x);
  LUABIND_RETURN(MatrixFloat, obj->logpdfDerivative(x, grads));
  DecRef(x);
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase clone
{
  LUABIND_RETURN(StatisticalDistributionBase, obj->clone());
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase size
{
  LUABIND_RETURN(uint, obj->getSize());
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase to_lua_string
{
  const char *format;
  bool is_ascii = false;
  LUABIND_GET_OPTIONAL_PARAMETER(1, string, format, "binary");
  if (strcmp(format,"ascii") == 0)
    is_ascii = true;
  else if (strcmp(format,"binary") != 0)
    LUABIND_FERROR1("Incorrect format, expected 'ascii' or 'binary', given '%s'",
                    format);
  char *str = obj->toLuaString(is_ascii);
  LUABIND_RETURN(string, str);
  delete[] str;
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME UniformDistribution stats.dist.uniform
//BIND_CPP_CLASS    UniformDistribution
//BIND_SUBCLASS_OF  UniformDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR UniformDistribution
{
  MatrixFloat *low, *high;
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, low);
    LUABIND_GET_PARAMETER(2, MatrixFloat, high);
  }
  else {
    float lowf, highf;
    LUABIND_GET_PARAMETER(1, float, lowf);
    LUABIND_GET_PARAMETER(2, float, highf);
    int dims[1] = { 1 };
    low  = new MatrixFloat(1, dims);
    high = new MatrixFloat(1, dims);
    (*low)(0)  = lowf;
    (*high)(0) = highf;
  }
  obj = new UniformDistribution(low, high);
  LUABIND_RETURN(UniformDistribution, obj);
}
//BIND_END

//BIND_METHOD UniformDistribution clone
{
  LUABIND_RETURN(UniformDistribution,
                 static_cast<UniformDistribution*>(obj->clone()));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME GeneralNormalDistribution stats.dist.normal.general
//BIND_CPP_CLASS    GeneralNormalDistribution
//BIND_SUBCLASS_OF  GeneralNormalDistribution StatisticalDistributionBase

//BIND_LUACLASSNAME DiagonalNormalDistribution stats.dist.normal.diagonal
//BIND_CPP_CLASS    DiagonalNormalDistribution
//BIND_SUBCLASS_OF  DiagonalNormalDistribution StatisticalDistributionBase

//BIND_LUACLASSNAME StandardNormalDistribution stats.dist.normal.standard
//BIND_CPP_CLASS    StandardNormalDistribution
//BIND_SUBCLASS_OF  StandardNormalDistribution StatisticalDistributionBase


//BIND_CONSTRUCTOR GeneralNormalDistribution
{
  LUABIND_ERROR("Use stats.dist.normal constructor");
}
//BIND_END

//BIND_CONSTRUCTOR DiagonalNormalDistribution
{
  LUABIND_ERROR("Use stats.dist.normal constructor");
}
//BIND_END

//BIND_CONSTRUCTOR StandardNormalDistribution
{
  LUABIND_ERROR("Use stats.dist.normal constructor");
}
//BIND_END

//BIND_FUNCTION stats.dist.normal
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  if (argn == 0) {
    StandardNormalDistribution *obj = new StandardNormalDistribution();
    LUABIND_RETURN(StandardNormalDistribution, obj);
  }
  else {
    MatrixFloat *mean;
    if (lua_isMatrixFloat(L,1)) {
      LUABIND_GET_PARAMETER(1, MatrixFloat, mean);
      if (lua_isMatrixFloat(L,2)) {
        MatrixFloat *cov;
        LUABIND_GET_PARAMETER(2, MatrixFloat, cov);
        GeneralNormalDistribution *obj = new GeneralNormalDistribution(mean, cov);
        LUABIND_RETURN(GeneralNormalDistribution, obj);
      }
      else {
        SparseMatrixFloat *cov;
        LUABIND_GET_PARAMETER(2, SparseMatrixFloat, cov);
        DiagonalNormalDistribution *obj = new DiagonalNormalDistribution(mean, cov);
        LUABIND_RETURN(DiagonalNormalDistribution, obj);
      }
    }
    else {
      SparseMatrixFloat *cov;
      float mu, sigma;
      LUABIND_GET_PARAMETER(1, float, mu);
      LUABIND_GET_PARAMETER(2, float, sigma);
      int dims[1] = { 1 };
      mean = new MatrixFloat(1, dims);
      (*mean)(0) = mu;
      cov = SparseMatrixFloat::diag(1, sigma);
      DiagonalNormalDistribution *obj = new DiagonalNormalDistribution(mean, cov);
      LUABIND_RETURN(DiagonalNormalDistribution, obj);
    }
  }
}
//BIND_END

//BIND_METHOD GeneralNormalDistribution clone
{
  LUABIND_RETURN(GeneralNormalDistribution,
                 static_cast<GeneralNormalDistribution*>(obj->clone()));
}
//BIND_END

//BIND_METHOD DiagonalNormalDistribution clone
{
  LUABIND_RETURN(DiagonalNormalDistribution,
                 static_cast<DiagonalNormalDistribution*>(obj->clone()));
}
//BIND_END

//BIND_METHOD StandardNormalDistribution clone
{
  LUABIND_RETURN(StandardNormalDistribution,
                 static_cast<StandardNormalDistribution*>(obj->clone()));
}
//BIND_END


//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME GeneralLogNormalDistribution stats.dist.lognormal.general
//BIND_CPP_CLASS    GeneralLogNormalDistribution
//BIND_SUBCLASS_OF  GeneralLogNormalDistribution StatisticalDistributionBase

//BIND_LUACLASSNAME DiagonalLogNormalDistribution stats.dist.lognormal.diagonal
//BIND_CPP_CLASS    DiagonalLogNormalDistribution
//BIND_SUBCLASS_OF  DiagonalLogNormalDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR GeneralNormalDistribution
{
  LUABIND_ERROR("Use stats.dist.lognormal constructor");
}
//BIND_END

//BIND_CONSTRUCTOR DiagonalNormalDistribution
{
  LUABIND_ERROR("Use stats.dist.lognormal constructor");
}
//BIND_END

//BIND_FUNCTION stats.dist.lognormal
{
  MatrixFloat *mean, *location;
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, mean);
    LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixFloat, location, 0);
    if (lua_isMatrixFloat(L,2)) {
      MatrixFloat *cov;
      LUABIND_GET_PARAMETER(2, MatrixFloat, cov);
      GeneralLogNormalDistribution *obj = new GeneralLogNormalDistribution(mean, cov,
                                                                           location);
      LUABIND_RETURN(GeneralLogNormalDistribution, obj);
    }
    else {
      SparseMatrixFloat *cov;
      LUABIND_GET_PARAMETER(2, SparseMatrixFloat, cov);
      DiagonalLogNormalDistribution *obj = new DiagonalLogNormalDistribution(mean, cov,
                                                                             location);
      LUABIND_RETURN(DiagonalLogNormalDistribution, obj);
    }
  }
  else {
    SparseMatrixFloat *cov;
    float mu, sigma, loc;
    LUABIND_GET_PARAMETER(1, float, mu);
    LUABIND_GET_PARAMETER(2, float, sigma);
    LUABIND_GET_OPTIONAL_PARAMETER(3, float, loc, 0.0f);
    int dims[1] = { 1 };
    mean = new MatrixFloat(1, dims);
    (*mean)(0) = mu;
    cov = SparseMatrixFloat::diag(1, sigma);
    if (loc != 0.0f) {
      location = new MatrixFloat(1, dims);
      (*location)(0) = loc;
    }
    else location = 0;
    DiagonalLogNormalDistribution *obj = new DiagonalLogNormalDistribution(mean, cov,
                                                                           location);
    LUABIND_RETURN(DiagonalLogNormalDistribution, obj);
  }
}
//BIND_END

//BIND_METHOD GeneralLogNormalDistribution clone
{
  LUABIND_RETURN(GeneralLogNormalDistribution,
                 static_cast<GeneralLogNormalDistribution*>(obj->clone()));
}
//BIND_END

//BIND_METHOD DiagonalLogNormalDistribution clone
{
  LUABIND_RETURN(DiagonalLogNormalDistribution,
                 static_cast<DiagonalLogNormalDistribution*>(obj->clone()));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ExponentialDistribution stats.dist.exponential
//BIND_CPP_CLASS    ExponentialDistribution
//BIND_SUBCLASS_OF  ExponentialDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR ExponentialDistribution
{
  MatrixFloat *lambda;
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, lambda);
  }
  else {
    float lambdaf;
    LUABIND_GET_PARAMETER(1, float, lambdaf);
    int dims[1] = {1};
    lambda = new MatrixFloat(1, dims);
    AprilMath::MatrixExt::Operations::matFill(lambda, lambdaf);
  }
  obj = new ExponentialDistribution(lambda);
  LUABIND_RETURN(ExponentialDistribution, obj);
}
//BIND_END

//BIND_METHOD ExponentialDistribution clone
{
  LUABIND_RETURN(ExponentialDistribution,
                 static_cast<ExponentialDistribution*>(obj->clone()));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BetaDistribution stats.dist.beta
//BIND_CPP_CLASS    BetaDistribution
//BIND_SUBCLASS_OF  BetaDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR BetaDistribution
{
  MatrixFloat *alpha, *beta;
  if (lua_isMatrixFloat(L, 1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, alpha);
    LUABIND_GET_PARAMETER(2, MatrixFloat, beta);
  }
  else {
    float alphaf, betaf;
    LUABIND_GET_PARAMETER(1, float, alphaf);
    LUABIND_GET_PARAMETER(2, float, betaf);
    int dims[1] = {1};
    alpha = new MatrixFloat(1, dims);
    (*alpha)(0) = alphaf;
    beta = new MatrixFloat(1, dims);
    (*beta)(0) = betaf;
  }
  obj = new BetaDistribution(alpha, beta);
  LUABIND_RETURN(BetaDistribution, obj);
}
//BIND_END

//BIND_METHOD BetaDistribution clone
{
  LUABIND_RETURN(BetaDistribution,
                 static_cast<BetaDistribution*>(obj->clone()));
}
//BIND_END


//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BinomialDistribution stats.dist.binomial
//BIND_CPP_CLASS    BinomialDistribution
//BIND_SUBCLASS_OF  BinomialDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR BinomialDistribution
{
  MatrixFloat *n, *p;
  if (lua_isMatrixFloat(L, 1)) {
    LUABIND_GET_PARAMETER(1, MatrixFloat, n);
    LUABIND_GET_PARAMETER(2, MatrixFloat, p);
  }
  else {
    unsigned int ni;
    float pf;
    LUABIND_GET_PARAMETER(1, uint, ni);
    LUABIND_GET_PARAMETER(2, float, pf);
    int dims[1] = {1};
    n = new MatrixFloat(1, dims);
    (*n)(0) = static_cast<float>(ni);
    p = new MatrixFloat(1, dims);
    (*p)(0) = pf;
  }
  obj = new BinomialDistribution(n, p);
  LUABIND_RETURN(BinomialDistribution, obj);
}
//BIND_END

//BIND_METHOD BinomialDistribution clone
{
  LUABIND_RETURN(BinomialDistribution,
                 static_cast<BinomialDistribution*>(obj->clone()));
}
//BIND_END
