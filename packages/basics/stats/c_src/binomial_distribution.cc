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
#include <ctgmath> // FIXME: this library needs c++11 extensions
#include "buffer_list.h"
#include "error_print.h"
#include "binomial_distribution.h"
#include "combinations.h"
#include "utilMatrixFloat.h"

using april_utils::buffer_list;
using april_utils::log_float;
using basics::MatrixFloat;
using basics::MTRand;

namespace Stats {
  
  BinomialDistribution::BinomialDistribution(MatrixFloat *n, MatrixFloat *p) :
    StatisticalDistributionBase(1),
    n(n), p(p) {
    if (n->getNumDim() != 1 || p->getNumDim() != 1 ||
        n->getDimSize(0) != 1 || p->getDimSize(0) != 1)
      ERROR_EXIT(128, "Expected n,p one-dimensional matrices with size 1\n");
    updateParams();
  }

  BinomialDistribution::~BinomialDistribution() {
    DecRef(n);
    DecRef(p);
  }
  
  void BinomialDistribution::updateParams() {
    float nf = (*n)(0);
    pf = (*p)(0);
    //
    if (!(nf > 0.0f) || !(pf > 0.0f))
      ERROR_EXIT(128, "Binomial distribution needs > 0 n and p params\n");
    if (!(pf < 1.0f))
      ERROR_EXIT(128, "Binomial distribution needs < 1 p param\n");
    if (floorf(nf) != nf)
      ERROR_EXIT(128, "Binomial distribution needs an integer n param\n");
    //
    pfm1 = 1.0f - pf;
    if (pf == 1.0f) {
      lpf = log_float::one();
      lpfm1 = log_float::zero();
    }
    else if (pf == 0.0f) {
      lpf = log_float::zero();
      lpfm1 = log_float::one();
    }
    else {
      lpf = log_float::from_float(pf);
      lpfm1 = log_float::from_float(pfm1);
    }
    ni = static_cast<int>(nf);
  }
  
  void BinomialDistribution::privateSample(MTRand *rng,
                                           MatrixFloat *result) {
    // generation via gamma variate
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end(); ++result_it) {
      // simulate n Bernoulli trials, and sum all values
      int counts = 0;
      for (int i=0; i<ni; ++i) {
        if (rng->rand() < pf) ++counts;
      }
      *result_it = static_cast<float>(counts);
    }
  }
  
  log_float BinomialDistribution::computeDensity(int k) {
    unsigned int Ci = Combinations::get(ni, k);
    log_float C = log_float::from_float(static_cast<float>(Ci));
    return ( C *
             lpf.raise_to(static_cast<float>(k)) *
             lpfm1.raise_to(static_cast<float>(ni - k)) );
  }

  void BinomialDistribution::privateLogpdf(const MatrixFloat *x,
                                           MatrixFloat *result) {
    MatrixFloat::const_iterator x_it(x->begin());
    MatrixFloat::iterator result_it(result->begin());
    while(x_it != x->end()) {
      if (floorf(*x_it) != *x_it)
        ERROR_EXIT(128, "All values must be integers\n");
      if (*x_it < 0.0f || *x_it > static_cast<float>(ni))
        *result_it = log_float::zero().log();
      else {
        unsigned int k  = static_cast<unsigned int>(*x_it);
        *result_it = computeDensity(k).log();
      }
      ++x_it;
      ++result_it;
    }
  }

  void BinomialDistribution::privateLogcdf(const MatrixFloat *x,
                                           MatrixFloat *result) {
    MatrixFloat::const_iterator x_it(x->begin());
    MatrixFloat::iterator result_it(result->begin());
    while(x_it != x->end()) {
      if (floorf(*x_it) != *x_it)
        ERROR_EXIT(128, "All values must be integers\n");
      if (*x_it < 0.0f)
        *result_it = log_float::zero().log();
      else if (*x_it > static_cast<float>(ni))
        *result_it = log_float::one().log();
      else {
        log_float r = computeDensity(0);
        int k = static_cast<int>(*x_it);
        for (int i=1; i<=k; ++i) {
          r += computeDensity(i);
        }
        *result_it = r.log();
      }
      ++x_it;
      ++result_it;
    }
  }

  StatisticalDistributionBase *BinomialDistribution::clone() {
    return new BinomialDistribution(n->clone(), p->clone());
  }
  
  char *BinomialDistribution::toLuaString(bool is_ascii) const {
    UNUSED_VARIABLE(is_ascii);
    buffer_list buffer;
    buffer.printf("stats.dist.binomial(%d, %g)", ni, pf);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
