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
#ifndef FEATURE_BASED_LM_H
#define FEATURE_BASED_LM_H

#include <stdint.h>
#include "april_assert.h"
#include "bunch_hashed_LM.h"
#include "error_print.h"
#include "function_interface.h"
#include "history_based_LM.h"
#include "logbase.h"
#include "token_matrix.h"
#include "trie_vector.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {

  using april_utils::vector;

  template <typename Key, typename Score>
  class FeatureBasedLM;

  template <typename Key, typename Score>
  class FeatureBasedLMInterface : public HistoryBasedLMInterface <Key,Score>, public BunchHashedLMInterface <Key, Score> {
    friend class FeatureBasedLM<Key,Score>;

    Functions::FunctionInterface *filter;

  protected:
    FeatureBasedLMInterface(FeatureBasedLM<Key,Score>* model) :
      HistoryBasedLMInterface<Key,Score>(model),
      BunchHashedLMInterface<Key,Score>(model) {
      filter = model->getFilter();
      IncRef(filter);
    }

    typedef typename BunchHashedLMInterface<Key,Score>::KeyWordHash KeyWordHash;
    typedef typename BunchHashedLMInterface<Key,Score>::WordResultHash WordResultHash;

    virtual void executeQueries(Token *input) = 0;

    virtual void computeKeysAndScores(KeyWordHash &ctxt_hash,
                                      unsigned int bunch_size) {
      UNUSED_VARIABLE(bunch_size);
      FloatMatrix mat;
      Token *input = new TokenMatrixFloat(mat);
      // For each context key entry
      for (typename KeyWordHash::iterator it = ctxt_hash.begin();
        it != ctxt_hash.end(); ++it) {
        Key context_key = it->first;
        WordResultHash &word_hash = it->second;

        // For each word entry
        for (typename WordResultHash::iterator it2 = word_hash.begin();
          it2 != word_hash.end(); ++it2) {
          WordType word = it2->first;

          // Add pattern to token input with key and word
        }
      }
      Token *filtered_input = filter->calculate(input);
      executeQueries(filtered_input);
    }

  public:
    virtual ~FeatureBasedLMInterface() {
      DecRef(filter);
    }

    void incRef() {
      HistoryBasedLMInterface<Key,Score>::incRef();
      BunchHashedLMInterface<Key,Score>::incRef();
    }

    bool decRef() {
      HistoryBasedLMInterface<Key,Score>::decRef();
      BunchHashedLMInterface<Key,Score>::decRef();
      return (HistoryBasedLMInterface<Key,Score>::getRef() <= 0);
    }

    Token* applyFilter(Token* token) {
      return filter->calculate(token);
    }
  };

  template <typename Key, typename Score>
  class FeatureBasedLM : public HistoryBasedLM <Key,Score>, public BunchHashedLM <Key,Score> {
  private:
    Functions::FunctionInterface *filter;

  public:
    FeatureBasedLM(int ngram_order,
                   WordType init_word,
                   april_utils::TrieVector *trie_vector,
                   unsigned int bunch_size,
                   Functions::FunctionInterface *filter) :
      HistoryBasedLM<Key,Score>(ngram_order,
                                init_word,
                                trie_vector),
      BunchHashedLM<Key,Score>(bunch_size),
      filter(filter) {
      IncRef(filter);
    }

    virtual ~FeatureBasedLM() {
      DecRef(filter);
    }

    Functions::FunctionInterface* getFilter() {
      return filter;
    }

    void incRef() {
      HistoryBasedLM<Key,Score>::incRef();
      BunchHashedLM<Key,Score>::incRef();
    }

    bool decRef() {
      HistoryBasedLM<Key,Score>::decRef();
      BunchHashedLM<Key,Score>::decRef();
      return (HistoryBasedLM<Key,Score>::getRef() <= 0);
    }
  };

  typedef FeatureBasedLMInterface<uint32_t, log_float> FeatureBasedLMInterfaceUInt32LogFloat;
  typedef FeatureBasedLM<uint32_t, log_float> FeatureBasedLMUInt32LogFloat;
}; // closes namespace

#endif // FEATURE_BASED_LM_H
