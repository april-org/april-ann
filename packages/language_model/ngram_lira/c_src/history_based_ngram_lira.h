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

#ifndef HISTORY_BASED_NGRAM_LIRA_H
#define HISTORY_BASED_NGRAM_LIRA_H

#include "history_based_LM.h"
#include "ngram_lira.h"

namespace LanguageModels {

  class HistoryBasedNgramLiraLM;
  
  class HistoryBasedNgramLiraLMInterface :
    public HistoryBasedLMInterface< NgramLiraModel::Key,
                                    NgramLiraModel::Score > {
  public:
    typedef NgramLiraModel::Key Key;
    typedef NgramLiraModel::Score Score;
    
    virtual ~HistoryBasedNgramLiraLMInterface();

    virtual BasicArcsIterator beginBasicArcs(Key key, Score threshold) {
      return lira_interface->beginBasicArcs(key, threshold);
    }
    
    virtual BasicArcsIterator endBasicArcs(Key key) {
      return lira_interface->endBasicArcs(key);
    }

  protected:
    friend class HistoryBasedNgramLiraLM;
    HistoryBasedNgramLiraLMInterface(HistoryBasedNgramLiraLM *model,
				     NgramLiraModel *lira_model);

    virtual bool privateBestProb(Key k,
                                 const WordType *context_words,
                                 unsigned int context_size,
                                 Score &score);

    virtual Score privateBestProb() const;
    
    virtual bool privateGetFinalScore(Key key,
                                      const WordType *context_words,
                                      unsigned int context_size,
                                      Score threshold,
                                      Score &score);

    virtual bool privateGet(Key key,
                            WordType word,
                            const WordType *context_words,
                            unsigned int context_size,
                            Score threshold,
                            Score &score);
    
  private:

    NgramLiraInterface *lira_interface;
  
  };

  class HistoryBasedNgramLiraLM : public HistoryBasedLM<NgramLiraModel::Key,
                                                        NgramLiraModel::Score > {
  public:
    typedef NgramLiraModel::Key Key;
    typedef NgramLiraModel::Score Score;

  private:
    NgramLiraModel *lira_model;

  public:
    HistoryBasedNgramLiraLM(WordType init_word,
                            AprilUtils::TrieVector *trie_vector,
                            NgramLiraModel *lira_model);

    virtual ~HistoryBasedNgramLiraLM();

    virtual LMInterface<Key,Score>* getInterface();
  };

} // closes namespace language_models

#endif // HISTORY_BASED_NGRAM_LIRA_H
