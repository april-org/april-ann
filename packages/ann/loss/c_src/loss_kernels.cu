/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Salvador España-Boquera, Francisco Zamora-Martinez, Adrián
 * Palacios-Corella
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
#include "cmath_overloads.h"
#include "loss_kernels.h"
#include "map_matrix.h"
#include "reduce_matrix.h"
#include "smart_ptr.h"

using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;

namespace AprilMath {
  namespace MatrixExt {
    namespace LossOperations {
      
      namespace Kernels {

        struct MSE {
          APRIL_CUDA_EXPORT float operator()(const float &input,
                                             const float &target) const {
            float diff = input - target;
            return 0.5f * diff * diff;
          }
        };
        
        /////////////////////////////////////////////////////////////////////////
        
        struct CrossEntropy {
          const float log_epsilon, log_1_epsilon, EPSILON;
          
          CrossEntropy(float EPSILON) : log_epsilon(logf(EPSILON)),
                                        log_1_epsilon(logf(1.0f - EPSILON)),
                                        EPSILON(EPSILON) {
          }
          
          APRIL_CUDA_EXPORT float operator()(const float &input,
                                             const float &target) const {
            // TODO: check errors using GPU/CPU error capture facility
            /*
              april_assert(!(input > 0.0f) &&
              "Only log-based activation functions are allowed");
              april_assert(!(target < 0.0f) && !(target > 1.0f) &&
              "Only [0,1] target patterns are allowed");
            */
            // compute derivative
            float  log_o     = m_clamp(input, log_epsilon, log_1_epsilon);
            double o         = m_exp(log_o);
            float  log_inv_o = m_log(1.0 - o);
            // CLAMP of reference (target)
            float  t         = m_clamp(target, EPSILON, 1.0f - EPSILON);
            // CLAMP of 1.0 - reference (target). We do clamp again to avoid
            // numerical approximation problems, and to ensure correct working of
            // inv_t > EPSILON comparison
            float  inv_t     = m_clamp(1.0f - target, EPSILON, 1.0f - EPSILON);
            // printf("%g * %g + %g * %g :: %g\n", t, log_o, inv_t, log_inv_o, o);
            float sum;
            if (t > EPSILON) sum = -t * log_o;
            else sum = 0.0f;
            if (inv_t > EPSILON) sum -= inv_t * log_inv_o;
            return sum;
          }
        };

        /////////////////////////////////////////////////////////////////////////

        struct NonPairedCrossEntropy {
          const float EPSILON;
          
          NonPairedCrossEntropy(float EPSILON) : EPSILON(EPSILON) {
          }
          
          APRIL_CUDA_EXPORT float operator()(const float &input,
                                             const float &target) const {
            // TODO: check errors using GPU/CPU error capture facility
            /*
              april_assert(!(input > 0.0f) &&
              "Only log-based activation functions are allowed");
              april_assert(!(target < 0.0f) && !(target > 1.0f) &&
              "Only [0,1] target patterns are allowed");
            */
            // compute derivative
            float o     = m_clamp(input, EPSILON, 1.0f - EPSILON);
            float log_o = m_log(o);
            float log_inv_o = m_log(1.0f - o);
            // CLAMP of reference (target)
            float  t = m_clamp(target, EPSILON, 1.0f - EPSILON);
            // CLAMP of 1.0 - reference (target). We do clamp again to avoid
            // numerical approximation problems, and to ensure correct working of
            // inv_t > EPSILON comparison
            float  inv_t = m_clamp(1.0f - target, EPSILON, 1.0f - EPSILON);
            // printf("%g * %g + %g * %g :: %g\n", t, log_o, inv_t, log_inv_o, o);
            float sum;
            if (t > EPSILON) sum = -t * log_o;
            else sum = 0.0f;
            if (inv_t > EPSILON) sum -= inv_t * log_inv_o;
            return sum;
          }
        };

        /////////////////////////////////////////////////////////////////////////
        
        struct CrossEntropyGradient {
          const float log_epsilon, log_1_epsilon, EPSILON;
          
          CrossEntropyGradient(float EPSILON) : log_epsilon(logf(EPSILON)),
                                                log_1_epsilon(logf(1.0f - EPSILON)),
                                                EPSILON(EPSILON) {
          }
          
          APRIL_CUDA_EXPORT float operator()(const float &input,
                                             const float &target) const {
            return m_exp(m_clamp(input, log_epsilon, log_1_epsilon)) - target;
          }
        };

        struct NonPairedCrossEntropyGradient {
          const float EPSILON;
          
          NonPairedCrossEntropyGradient(float EPSILON) : EPSILON(EPSILON) {
          }
          
          APRIL_CUDA_EXPORT float operator()(const float &input,
                                             const float &target) const {
            float o = m_clamp(input,  EPSILON, 1.0f - EPSILON);
            float t = m_clamp(target, EPSILON, 1.0f - EPSILON);
            return -(t/o - (1.0f - t)/(1.0f - o));
          }
        };
        
        /////////////////////////////////////////////////////////////////////////

        struct MAEGradient {
          const float EPSILON, invN;
          
          MAEGradient(float &EPSILON, int invN) : EPSILON(EPSILON),
                                                  invN(invN) { }
          
          APRIL_CUDA_EXPORT float operator()(const float &input,
                                             const float &target) const {
            float d = input - target;
            if (m_abs(d) < EPSILON) d = 0.0f;
            else {
              if (d < 0.0f) d = -invN;
              else d = invN;
            }
            return d;
          }
        };
        
        /////////////////////////////////////////////////////////////////////////

        struct MultiClassCrossEntropy {
          const float EPSILON;
          
          MultiClassCrossEntropy(float EPSILON) : EPSILON(EPSILON) { }
          
          APRIL_CUDA_EXPORT float operator()(const float &input, const float &target) const {
            float log_o = input;
            float t = m_clamp(target, EPSILON, 1.0f - EPSILON);
            float sum;
            if (t > EPSILON) sum = -t * log_o;
            else sum = 0.0f;
            return sum;
          }
        };
        
        /////////////////////////////////////////////////////////////////////////
        
      } // namespace Kernels

      void matMSE(Basics::MatrixFloat *output,
                  const Basics::MatrixFloat *input,
                  const Basics::MatrixFloat *target) {
        AprilUtils::SharedPtr<Basics::MatrixFloat>
          map_output( MatrixScalarMap2(input, target, Kernels::MSE(),
                                       input->cloneOnlyDims()) );
        matSum(map_output.get(), 1, output);
      }
      
      void matCrossEntropy(Basics::MatrixFloat *output,
                           const Basics::MatrixFloat *input,
                           const Basics::MatrixFloat *target,
                           float near_zero) {
        Kernels::CrossEntropy cross_entropy(near_zero);
        AprilUtils::SharedPtr<Basics::MatrixFloat>
          map_output(MatrixScalarMap2(input, target, cross_entropy,
                                      input->cloneOnlyDims()));
        matSum(map_output.get(), 1, output);
      }

      void matNonPairedCrossEntropy(Basics::MatrixFloat *output,
                                    const Basics::MatrixFloat *input,
                                    const Basics::MatrixFloat *target,
                                    float near_zero) {
        Kernels::NonPairedCrossEntropy cross_entropy(near_zero);
        AprilUtils::SharedPtr<Basics::MatrixFloat>
          map_output(MatrixScalarMap2(input, target, cross_entropy,
                                      input->cloneOnlyDims()));
        matSum(map_output.get(), 1, output);
      }      
      /////////////////////////////////////////////////////////////////////////

      void matCrossEntropyGradient(Basics::MatrixFloat *output,
                                   const Basics::MatrixFloat *input,
                                   const Basics::MatrixFloat *target,
                                   float near_zero) {
        Kernels::CrossEntropyGradient cross_entropy_gradient(near_zero);
        MatrixScalarMap2(input, target, cross_entropy_gradient, output);
      }

      void matNonPairedCrossEntropyGradient(Basics::MatrixFloat *output,
                                            const Basics::MatrixFloat *input,
                                            const Basics::MatrixFloat *target,
                                            float near_zero) {
        Kernels::NonPairedCrossEntropyGradient cross_entropy_gradient(near_zero);
        MatrixScalarMap2(input, target, cross_entropy_gradient, output);
      }

      /////////////////////////////////////////////////////////////////////////

      void matMAEGradient(Basics::MatrixFloat *output,
                          const Basics::MatrixFloat *input,
                          const Basics::MatrixFloat *target,
                          float near_zero,
                          float inv_n) {
        Kernels::MAEGradient mae_gradient(near_zero, inv_n);
        MatrixScalarMap2(input, target, mae_gradient, output);
      }
      
      /////////////////////////////////////////////////////////////////////////
      
      void matMultiClassCrossEntropy(Basics::MatrixFloat *output,
                                     const Basics::MatrixFloat *input,
                                     const Basics::MatrixFloat *target,
                                     float near_zero) {
        Kernels::MultiClassCrossEntropy multi_class_cross_entropy(near_zero);
        AprilUtils::SharedPtr<Basics::MatrixFloat>
          map_output(MatrixScalarMap2(input, target, multi_class_cross_entropy,
                                      input->cloneOnlyDims()));
        matSum(map_output.get(), 1, output);
      }

      /////////////////////////////////////////////////////////////////////////
    } // namespace LossOperations
  } // namespace MatrixExt
} // namespace AprilMath
