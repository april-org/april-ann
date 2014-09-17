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

#ifndef BUNCH_HASHED_NGRAM_LIRA_H
#define BUNCH_HASHED_NGRAM_LIRA_H

#include "bunch_hashed_LM.h"
#include "ngram_lira.h"

namespace LanguageModels {

  class BunchHashedNgramLiraLM;
  
  class BunchHashedNgramLiraLMInterface :
    public BunchHashedLMInterface< NgramLiraModel::Key,
                                   NgramLiraModel::Score > {
  public:
    typedef NgramLiraModel::Key Key;
    typedef NgramLiraModel::Score Score;
    
    virtual ~BunchHashedNgramLiraLMInterface();
    virtual Score getBestProb() const;
    virtual Score getBestProb(Key k);
    virtual Score getFinalScore(Key k, Score threshold);
    virtual bool getZeroKey(Key &k) const;
    virtual Key getInitialKey();
    virtual void get(Key key, WordType word, Burden burden,
                     AprilUtils::vector<KeyScoreBurdenTuple> &result,
                     Score threshold);
  protected:
    friend class BunchHashedNgramLiraLM;
    BunchHashedNgramLiraLMInterface(BunchHashedNgramLiraLM *model,
				     NgramLiraModel *lira_model);

    virtual void computeKeysAndScores(KeyWordHash &ctxt_hash,
                                      unsigned int bunch_size);
  private:

    NgramLiraInterface *lira_interface;
  
  };

  class BunchHashedNgramLiraLM : public BunchHashedLM<NgramLiraModel::Key,
                                                      NgramLiraModel::Score > {
  public:
    typedef NgramLiraModel::Key Key;
    typedef NgramLiraModel::Score Score;

  private:
    NgramLiraModel *lira_model;

  public:
    BunchHashedNgramLiraLM(unsigned int bunch_size,
                           NgramLiraModel *lira_model);

    virtual ~BunchHashedNgramLiraLM();

    virtual LMInterface<Key,Score>* getInterface();
  };

} // closes namespace language_models

#endif // BUNCH_HASHED_NGRAM_LIRA_H
