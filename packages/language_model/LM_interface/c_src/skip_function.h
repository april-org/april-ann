/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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

namespace Functions {
  
  class SkipFunction : public FunctionInterface {
    Basics::Dice *dice;
    Basics::MTRand *random;
    LanguageModels::WordType mask_value;
  public:
    SkipFunction(Basics::Dice *dice, Basics::MTRand *random, LanguageModels::WordType mask_value) :
      FunctionInterface(),
      dice(dice),
      random(random),
      mask_value(mask_value) {
      IncRef(dice);
      IncRef(random);
    }

    virtual ~SkipFunction() {
      DecRef(dice);
      DecRef(random);
    }
    /// It returns the input (or domain) size of the function.
    virtual unsigned int getInputSize() const {
      return 0;
    }
    /// It returns the output (or range) size of the function.
    virtual unsigned int getOutputSize() const {
      return 0;
    }
    virtual Basics::Token *calculate(Basics::Token *input) {
      if (input->getTokenCode() != Basics::table_of_token_codes::vector_Tokens)
        ERROR_EXIT(128, "Input token should be a collection of tokens!\n");
      Basics::TokenBunchVector *bunch_of_tokens = input->convertTo<Basics::TokenBunchVector*>();

      for (unsigned int i = 0; i < bunch_of_tokens->size(); i++) {
        if ((*bunch_of_tokens)[i]->getTokenCode() != Basics::table_of_token_codes::vector_uint32)
          ERROR_EXIT(128, "Tokens from input token should be a collection of uint tokens!\n");
        Basics::TokenVectorUint32 *word_tokens = (*bunch_of_tokens)[i]->convertTo<Basics::TokenVectorUint32*>();
        int size = word_tokens->size();
        int skip_mask = dice->thrown(random);
        for (int j = size-1; j >= 0; j--) {
          if (skip_mask % 2)
            (*word_tokens)[j] = mask_value;
          skip_mask /= 2;
        }
      }
      return input;
    }
  };
}

#endif //SKIP_FUNCTION_H
