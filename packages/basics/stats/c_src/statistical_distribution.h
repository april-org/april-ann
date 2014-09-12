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
#ifndef STATISTICAL_DISTRIBUTION_H
#define STATISTICAL_DISTRIBUTION_H

#include "error_print.h"
#include "logbase.h"
#include "matrixFloat.h"
#include "matrixFloatSet.h"
#include "MersenneTwister.h"
#include "referenced.h"

namespace Stats {

  /**
   * @brief Class StatisticalDistributionBase implements basic interface for
   * statistical distributions and defines a private API for new classes
   * derivation.
   
   * All the methods work with col_major matrices (like ANN components), and
   * with two-dimensional matrices where first dimension is the bunch_size (like
   * ANN components).
   */
  class StatisticalDistributionBase : public Referenced {
    unsigned int size;
    
  protected:
    /**
     * @brief Receives a random generator and a MatrixFloat where sampled values
     * will be stored. The number of samples values will be equal to
     * result->getDimSize(0)
     *
     * @param rng - A MTRand instance.
     * @param result - A MatrixFloat where the sampled data will be stored.
     */
    virtual void privateSample(Basics::MTRand *rng,
                               Basics::MatrixFloat *result) = 0;
    /**
     * @brief Receives a x MatrixFloat with NxM size (N = bunch_size), and a N
     * sized result MatrixFloat.
     *
     * @param x - A NxM MatrixFloat with N data points of size M.
     * @param result - A MatrixFloat where logpdf will be stored.
     */
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result) = 0;
    /**
     * @brief Receives a x MatrixFloat with NxM size (N = bunch_size), and a N
     * sized result MatrixFloat.
     *
     * @param x - A NxM MatrixFloat with N data points of size M.
     * @param result - A MatrixFloat where logcdf will be stored.
     */
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result) = 0;
    /**
     * @brief Receives a x MatrixFloat with NxM size (N = bunch_size), and a N
     * sized grads MatrixFloat.
     *
     * @param x - A NxM MatrixFloat with N data points of size M.
     * @param result - A MatrixFloat where derivative of logpdf will be stored.
     */
    virtual void privateLogpdfDerivative(const Basics::MatrixFloat *x,
                                         Basics::MatrixFloat *result) {
      UNUSED_VARIABLE(x);
      UNUSED_VARIABLE(result);
      ERROR_EXIT(128, "Derivative not implemented\n");
    }
  public:
    /**
     * @param size - The size of the distribution data points.
     */
    StatisticalDistributionBase(unsigned int size) : Referenced(), size(size) {}
    virtual ~StatisticalDistributionBase() {}
    /**
     * @brief Public part of sample method, where arguments will be checked.
     *
     * @param rng - A MTRand instance.
     *
     * @param result - An auxiliary MatrixFloat where to store the result, if
     * not given a MatrixFloat with 1xM size will be allocated.
     *
     * @return A MatrixFloat with the sample result.
     */
    Basics::MatrixFloat *sample(Basics::MTRand *rng,
                                Basics::MatrixFloat *result=0) {
      int dims[2] = { 1, static_cast<int>(size) };
      if (result == 0) {
        result = new Basics::MatrixFloat(2, dims, CblasColMajor);
      }
      else if (result->getNumDim() != 2 || result->getDimSize(1) != static_cast<int>(size))
        ERROR_EXIT1(128, "Incorrect result matrix size, expected "
                    "bi-dimensional matrix with Nx%u shape\n", size);
      else if (result->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major order in result matrix\n");
      // virtual call
      privateSample(rng, result);
      return result;
    }
    /**
     * @brief Public part of logpdf method, arguments will be checked here.
     *
     * @param x - A MatrixFloat with size NxM where N is the number of data
     * points.
     *
     * @param result - An auxiliary MatrixFloat where to store the result.
     *
     * @return A MatrixFloat with the sample result.
     */
    Basics::MatrixFloat *logpdf(const Basics::MatrixFloat *x,
                                Basics::MatrixFloat *result=0) {
      if (x->getNumDim() != 2 || x->getDimSize(1) != static_cast<int>(size))
        ERROR_EXIT1(128, "Incorrect x matrix size, expected bi-dimensional "
                    "matrix with Nx%u shape\n", size);
      if (x->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major in x matrix\n");
      int dims[1] = { x->getDimSize(0) };
      if (result == 0) {
        result = new Basics::MatrixFloat(1, dims, CblasColMajor);
      }
      else if (result->getNumDim() != 1 || result->getDimSize(0) != dims[0])
        ERROR_EXIT1(128, "Incorrect result matrix size, expected "
                    "one-dimensional matrix with %d size\n", dims[0]);
      else if (result->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major order in result matrix\n");
      // virtual call
      privateLogpdf(x, result);
      return result;
    }
    /**
     * @brief Public part of logcdf method, arguments will be checked here.
     *
     * @param x - A MatrixFloat with size NxM where N is the number of data
     * points.
     *
     * @param result - An auxiliary MatrixFloat where to store the result.
     *
     * @return A MatrixFloat with the sample result.
     */
    Basics::MatrixFloat *logcdf(const Basics::MatrixFloat *x,
                                Basics::MatrixFloat *result=0) {
      if (x->getNumDim() != 2 || x->getDimSize(1) != static_cast<int>(size))
        ERROR_EXIT1(128, "Incorrect x matrix size, expected bi-dimensional "
                    "matrix with Nx%u shape\n", size);
      if (x->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major in x matrix\n");
      int dims[1] = { x->getDimSize(0) };
      if (result == 0) {
        result = new Basics::MatrixFloat(1, dims, CblasColMajor);
      }
      else if (result->getNumDim() != 1 || result->getDimSize(0) != dims[0])
        ERROR_EXIT1(128, "Incorrect result matrix size, expected "
                    "one-dimensional matrix with %d size\n", dims[0]);
      else if (result->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major order in result matrix\n");
      // virtual call
      privateLogcdf(x, result);
      return result;
    }
    /**
     * @brief Public part of logpdfDerivative method, arguments will be checked
     * here.
     *
     * @param x - A MatrixFloat with size NxM where N is the number of data
     * points.
     *
     * @param result - An auxiliary MatrixFloat where to store the result.
     *
     * @return A MatrixFloat with the sample result.
     */
    Basics::MatrixFloat *logpdfDerivative(const Basics::MatrixFloat *x,
                                          Basics::MatrixFloat *grads=0) {
      if (x->getNumDim() != 2 || x->getDimSize(1) != static_cast<int>(size))
        ERROR_EXIT1(128, "Incorrect x matrix size, expected bi-dimensional "
                    "matrix with Nx%u shape\n", size);
      if (x->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major in x matrix\n");
      const int *dims = x->getDimPtr();
      if (grads == 0) {
        grads = new Basics::MatrixFloat(2, dims, CblasColMajor);
        AprilMath::MatrixExt::Operations::matZeros(grads);
      }
      else if (grads->getNumDim() != 2 || grads->getDimSize(0) != dims[0] ||
               grads->getDimSize(1) != dims[1])
        ERROR_EXIT2(128, "Incorrect grads matrix size, expected "
                    "bidimensional matrix with %dx%d size\n", dims[0], dims[1]);
      else if (grads->getMajorOrder() != CblasColMajor)
        ERROR_EXIT(128, "Expected col_major order in grads matrix\n");
      // virtual call
      IncRef(grads);
      privateLogpdfDerivative(x, grads);
      // WARNING: don't use DecRef macro because it will destroy grads matrix
      grads->decRef();
      return grads;
    }
    /**
     * @return The size of data points.
     */
    unsigned int getSize() { return size; }
    
    // abstract interface
    
    /**
     * @return A deep copy (clone) of the caller object.
     */
    virtual StatisticalDistributionBase *clone() = 0;
    /**
     * @param is_ascii - A bool indicating if the Matrix data will be stored in
     * ascii (or binary) format.
     *
     * @return A Lua string representation of the caller object, useful for
     * serialization purposes.
     */
    virtual char *toLuaString(bool is_ascii) const = 0;
  };
  
  ////////////////////////////////////////////////////////////////////////////
  /*

    class ExponentialDistribution : public StatisticalDistributionBase {
    MatrixFloat *lambda;
    };

    class DiagonalNormalDistribution : public StatisticalDistributionBase {
    MatrixFloat *mean;
    SparseMatrixFloat *cov;
    };

    class GeneralNormalDistribution : public StatisticalDistributionBase {
    MatrixFloat *mean, *cov;
    };

    class StudentDistribution : public StatisticalDistributionBase {
    MatrixFloat *v;
    };
  
    class GammaDistribution : public StatisticalDistributionBase {
    MatrixFloat *alpha, *beta;
    };
  
    class BetaDistribution : public StatisticalDistributionBase {
    MatrixFloat *alpha, *beta;
    };

    class BinomialDistribution : public StatisticalDistributionBase {
    MatrixFloat *p;
    };

    class MultinomialDistribution : public StatisticalDistributionBase {
    MatrixFloat *p;
    };

  */
  
}

#endif // STATISTICAL_DISTRIBUTION_H
