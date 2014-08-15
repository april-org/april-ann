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

#include "history_based_ngram_lira.h"
#include "unused_variable.h"

using namespace april_utils;

namespace LanguageModels {
  
  typedef NgramLiraModel::Key Key;
  typedef NgramLiraModel::Score Score;

  HistoryBasedNgramLiraLMInterface::
  HistoryBasedNgramLiraLMInterface(HistoryBasedNgramLiraLM *model,
				   NgramLiraModel *lira_model) :
    HistoryBasedLMInterface(model) {
    lira_interface = 
      static_cast<NgramLiraInterface*>(lira_model->getInterface());
    IncRef(lira_interface);
  }

  HistoryBasedNgramLiraLMInterface::
  ~HistoryBasedNgramLiraLMInterface() {
    DecRef(lira_interface);
  }

  bool HistoryBasedNgramLiraLMInterface::
  privateBestProb(Key k,
                  const WordType *context_words,
                  unsigned int context_size,
                  Score &score) {
    UNUSED_VARIABLE(k);
    Key lira_k = lira_interface->findKeyFromNgram(context_words, context_size);
    score = lira_interface->getBestProb(lira_k);
    return true;
  }
  
  bool HistoryBasedNgramLiraLMInterface::
  privateGetFinalScore(Key key,
                       const WordType *context_words,
                       unsigned int context_size,
                       Score threshold,
                       Score &score) {
    UNUSED_VARIABLE(key);
    UNUSED_VARIABLE(threshold);
    Key lira_k = lira_interface->findKeyFromNgram(context_words, context_size);
    score = lira_interface->getFinalScore(lira_k, Score::zero());
    return true;
  }
  
  bool HistoryBasedNgramLiraLMInterface::
  privateGet(Key key,
             WordType word,
             const WordType *context_words,
             unsigned int context_size,
             Score threshold,
             Score &score) {
    UNUSED_VARIABLE(key);
    UNUSED_VARIABLE(threshold);
    vector <KeyScoreBurdenTuple> result;
    Key st = lira_interface->findKeyFromNgram(context_words, context_size);
    lira_interface->get(st, word, Burden(-1, -1), result, Score::zero());
    score = result[0].key_score.score;
    return true;
  }
  
  Score HistoryBasedNgramLiraLMInterface::
  getBestProb() const {
    return lira_interface->getBestProb();
  }

  ////////////////////////////////////////////////////////////////////////////
  
  HistoryBasedNgramLiraLM::
  HistoryBasedNgramLiraLM(WordType init_word,
			  april_utils::TrieVector *trie_vector,
			  NgramLiraModel *lira_model) :
    HistoryBasedLM(lira_model->ngramOrder(), init_word, trie_vector),
    lira_model(lira_model) {
    IncRef(lira_model);
  }

  HistoryBasedNgramLiraLM::~HistoryBasedNgramLiraLM() {
    DecRef(lira_model);
  }
  
  LMInterface<Key,Score>* HistoryBasedNgramLiraLM::getInterface() {
    return new HistoryBasedNgramLiraLMInterface(this, lira_model);
  }
  
} // closes namespace language_models
