/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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

#ifndef SKIP_FUNCTION_H
#define SKIP_FUNCTION_H

#include "function_interface.h"
#include "dice.h"
#include "LM_interface.h"
#include "smart_ptr.h"
#include "token_vector.h"

namespace LanguageModels {
  namespace QueryFilters {
  
    /**
     * @brief Applies a stochastic skip mask using a Basics::Dice to a query
     * of a feature based language model.
     *
     * @note The dice size is \f$ 2^N \f$ being \f$N\f$ the number of expected
     * words.
     *
     * @note This function operates **in-place**
     */
    class DiceSkipFunction : public Functions::FunctionInterface {
      AprilUtils::SharedPtr<Basics::Dice> dice; ///< Binomial distribution.
      AprilUtils::SharedPtr<Basics::MTRand> random; ///< For stochastic purposes.
      LanguageModels::WordType mask_value; ///< The value used to replace masked words.
      unsigned int num_ctxt_words; ///< Number of expected words.
    public:
      /// Takes all given values.
      DiceSkipFunction(Basics::Dice *dice, Basics::MTRand *random,
                       LanguageModels::WordType mask_value);

      virtual ~DiceSkipFunction();

      /// Returns the number of expected words.
      virtual unsigned int getInputSize() const;

      /// Returns the number of expected words.
      virtual unsigned int getOutputSize() const;

      /**
       * @brief Applies the stochastic skip mask operation given a Basics::Token
       * instance.
       *
       * @see LanguageModels::FeatureBasedLM::query_filter property for
       * a description of the expected input.
       *
       * @note This function operates **in-place**
       */
      virtual Basics::Token *calculate(Basics::Token *input);
    };
    
  } // namespace QueryFilters
} // namespace LanguageModels

#endif //SKIP_FUNCTION_H
