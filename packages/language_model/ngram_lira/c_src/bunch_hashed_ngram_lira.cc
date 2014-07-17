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

#include "bunch_hashed_ngram_lira.h"
#include "unused_variable.h"

namespace LanguageModels {
  
  typedef NgramLiraModel::Key Key;
  typedef NgramLiraModel::Score Score;

  BunchHashedNgramLiraLMInterface::
  BunchHashedNgramLiraLMInterface(BunchHashedNgramLiraLM *model,
				   NgramLiraModel *lira_model) :
    BunchHashedLMInterface(model) {
    lira_interface = 
      static_cast<NgramLiraInterface*>(lira_model->getInterface());
    IncRef(lira_interface);
  }

  BunchHashedNgramLiraLMInterface::
  ~BunchHashedNgramLiraLMInterface() {
    DecRef(lira_interface);
  }

  Score BunchHashedNgramLiraLMInterface::
  getBestProb() const {
    return lira_interface->getBestProb();
  }

  Score BunchHashedNgramLiraLMInterface::
  getBestProb(Key k) {
    return lira_interface->getBestProb(k);
  }

  Score BunchHashedNgramLiraLMInterface::
  getFinalScore(Key k, Score threshold) { 
    return lira_interface->getFinalScore(k, threshold);
  }

  bool BunchHashedNgramLiraLMInterface::
  getZeroKey(Key &k) const {
    return lira_interface->getZeroKey(k);
  }

  Key BunchHashedNgramLiraLMInterface::
  getInitialKey() {
    return lira_interface->getInitialKey();
  }

  void BunchHashedNgramLiraLMInterface::
  get(Key key, WordType word, Burden burden,
      vector<KeyScoreBurdenTuple> &result,
      Score threshold) {
    lira_interface->get(key, word, burden, result, threshold);
  }

  void BunchHashedNgramLiraLMInterface::
  computeKeysAndScores(KeyWordHash &ctxt_hash,
                       unsigned int bunch_size) {
    UNUSED_VARIABLE(bunch_size);
    // For each context key entry
    for (typename KeyWordHash::const_iterator it = ctxt_hash.begin();
      it != ctxt_hash.end(); ++it) {
      Key context_key = it->first;
      WordResultHash word_hash = it->second;

      // For each word entry
      for (typename WordResultHash::const_iterator it2 = word_hash.begin();
        it2 != word_hash.end(); ++it2) {
        vector <KeyScoreBurdenTuple> tmp_result;
        WordType word = it2->first;
        KeyScoreMultipleBurdenTuple result_tuple = it2->second;
        lira_interface->get(context_key, word, Burden(-1, -1), tmp_result, Score::zero());
        if (tmp_result.size() != 1)
          ERROR_EXIT(127, "Multiple or zero results from lira!\n");
        // Debug
        //printf("KEY: %f\n", tmp_result[0].key_score.score);
        result_tuple.key_score.key = tmp_result[0].key_score.key;
        result_tuple.key_score.score = tmp_result[0].key_score.score;
      }
    }
  }


  ////////////////////////////////////////////////////////////////////////////
  
  BunchHashedNgramLiraLM::
  BunchHashedNgramLiraLM(unsigned int bunch_size,
                         NgramLiraModel *lira_model) :
    BunchHashedLM(lira_model->ngramOrder(), bunch_size),
    lira_model(lira_model) {
    IncRef(lira_model);
  }

  BunchHashedNgramLiraLM::~BunchHashedNgramLiraLM() {
    DecRef(lira_model);
  }
  
  LMInterface<Key,Score>* BunchHashedNgramLiraLM::getInterface() {
    return new BunchHashedNgramLiraLMInterface(this, lira_model);
  }
  
} // closes namespace language_models
