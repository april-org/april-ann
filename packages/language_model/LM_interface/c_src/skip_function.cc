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
#include "skip_function.h"

using Basics::Dice;
using Basics::MTRand;
using Basics::Token;
using Basics::TokenBunchVector;
using Basics::TokenVectorUint32;
using LanguageModels::WordType;

namespace Functions {
  
  DiceSkipFunction::DiceSkipFunction(Dice *dice, MTRand *random,
                                     WordType mask_value) :
    FunctionInterface(),
    dice(dice),
    random(random),
    mask_value(mask_value) {
    num_ctxt_words = 0;
    int aux = dice->getOutcomes();
    while(aux > 0) { aux >>= 1; ++num_ctxt_words; }
  }

  DiceSkipFunction::~DiceSkipFunction() { }

  unsigned int DiceSkipFunction::getInputSize() const {
    return num_ctxt_words;
  }
   
  unsigned int DiceSkipFunction::getOutputSize() const {
    return num_ctxt_words;
  }
  
  // Be careful, this function works **in-place**
  Token *DiceSkipFunction::calculate(Token *input) {
    if (input->getTokenCode() != Basics::table_of_token_codes::vector_Tokens) {
      ERROR_EXIT(128, "Input token should be a collection of tokens!\n");
    }
    TokenBunchVector *bunch_of_tokens = input->convertTo<TokenBunchVector*>();
    april_assert(bunch_of_tokens);
    
    // For every pattern in the given token.
    for (unsigned int i = 0; i < bunch_of_tokens->size(); i++) {
      if ((*bunch_of_tokens)[i]->getTokenCode() != Basics::table_of_token_codes::vector_Tokens) {
        ERROR_EXIT(128, "Tokens from input token should be a collection "
                   "with two tokens!\n");
      }
      TokenBunchVector *query_token =
        (*bunch_of_tokens)[i]->convertTo<TokenBunchVector*>();
      april_assert(query_token);
      
      if (query_token->size() != 2) {
        ERROR_EXIT2(128, "Expected %u tokens, found %u!\n",
                    2u, query_token->size());
      }
      
      if ((*query_token)[0]->getTokenCode() != Basics::table_of_token_codes::vector_uint32) {
        ERROR_EXIT(128, "Expected a collection of uint32_t!\n");
      }
      if ((*query_token)[1]->getTokenCode() != Basics::table_of_token_codes::vector_uint32) {
        ERROR_EXIT(128, "Expected a collection of uint32_t!\n");
      }
      
      TokenVectorUint32 *ctxt_words_token =
        (*query_token)[0]->convertTo<TokenVectorUint32*>();
      april_assert(ctxt_words_token);
      
      unsigned int size = ctxt_words_token->size();
      if (size != num_ctxt_words) {
        ERROR_EXIT2(128, "Incorrect number of words, expected %d, found %d\n",
                    num_ctxt_words, size);
      }
      
      // Compute a mask throwing the dice.
      int skip_mask = dice->thrown(random.get());
      for (int j = static_cast<int>(size)-1; j >= 0; j--) {
        if (skip_mask & 1) (*ctxt_words_token)[j] = mask_value;
        skip_mask >>= 1;
      }
    }
    
    return input;
  }

} // namespace Functions
