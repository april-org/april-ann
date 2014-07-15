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
#ifndef BUNCH_HASHED_LM_H
#define BUNCH_HASHED_LM_H

#include <stdint.h>
#include "hash_table.h"
#include "LM_interface.h"
#include "logbase.h"
#include "open_addressing_hash_with_fast_iterator.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {
  
  using april_utils::vector;
  using april_utils::open_addr_hash_fast_it;
  using april_utils::hash;
  
  template <typename Key, typename Score>
  class BunchHashedLM;
  
  /// BunchHashedLMInterface documentation ...
  template <typename Key, typename Score>
  class BunchHashedLMInterface : public LMInterface <Key,Score> {
    friend class BunchHashedLM<Key,Score>;
  protected:
    
    BunchHashedLMInterface(BunchHashedLM<Key,Score>* model) :
      LMInterface<Key,Score>(model) {
    }

    typedef typename LMInterface<Key,Score>::KeyScoreTuple KeyScoreTuple;
    typedef typename LMInterface<Key,Score>::KeyScoreBurdenTuple KeyScoreBurdenTuple;
    typedef typename LMInterface<Key,Score>::Burden Burden;

  public:
    struct KeyScoreMultipleBurdenTuple {
      KeyScoreTuple key_score;
      vector<Burden> burden_vector;
      KeyScoreMultipleBurdenTuple() {}
    };

    virtual ~BunchHashedLMInterface() {
    }
  protected:
    typedef hash<WordType, KeyScoreMultipleBurdenTuple> WordResultHash;
    typedef open_addr_hash_fast_it<Key, WordResultHash> KeyWordHash;

    virtual void computeKeysAndScores(KeyWordHash &ctxt_hash,
                                      unsigned int bunch_size) = 0;

  private:
    KeyWordHash context_key_hash;

  public:
    
    virtual void clearQueries() {
      LMInterface<Key,Score>::clearQueries();
      context_key_hash.clear();
    }

    virtual void insertQuery(Key key, WordType word, Burden burden,
                             Score threshold) {
      UNUSED_VARIABLE(threshold);
      WordResultHash &ctxt = context_key_hash[key];
      KeyScoreMultipleBurdenTuple &ctxt_word = ctxt[word];
      ctxt_word.burden_vector.push_back(burden);
    }

    virtual const vector<KeyScoreBurdenTuple> &getQueries() {
      BunchHashedLM<Key,Score> *mdl;
      mdl = static_cast<BunchHashedLM<Key,Score>*>(this->model);
      
      // compute keys and scores for queries in the hash table
      computeKeysAndScores(context_key_hash, mdl->getBunchSize());

      // For each context key entry
      for (typename KeyWordHash::const_iterator it = context_key_hash.begin();
        it != context_key_hash.end(); ++it) {
        Key context_key = it->first;
        WordResultHash word_hash = it->second;

        // For each word entry
        for (typename WordResultHash::const_iterator it2 = word_hash.begin();
          it2 != word_hash.end(); ++it2) {
          WordType word = it2->first;
          KeyScoreMultipleBurdenTuple result_tuple = it2->second;

          // For each burden at burden vector
          for (unsigned int i = 0; i < result_tuple.burden_vector.size(); ++i) {
            KeyScoreBurdenTuple new_tuple;

            new_tuple.key_score.key   = result_tuple.key_score.key;
            new_tuple.key_score.score = result_tuple.key_score.score;
            new_tuple.burden          = result_tuple.burden_vector[i];

            this->result.push_back(new_tuple);
          }
        }
      }
      return this->result;
    }
    

    /*
      virtual void get(Key key, WordType word, Burden burden,
                       vector<KeyScoreBurdenTuple> &result,
                       Score threshold) = 0;
    */
    // virtual void getNextKeys(Key key, WordType word, vector<Key> &result) = 0;
    // virtual bool getZeroKey(Key &k) const;
    // virtual Key getInitialKey();
    
  };
  
  template <typename Key, typename Score>
  class BunchHashedLM : public LMModel <Key,Score> {
  private:
    int ngram_order;
    unsigned int bunch_size;

  public:

    BunchHashedLM(int ngram_order,
                  unsigned int bunch_size) :
      LMModel<Key,Score>(),
      ngram_order(ngram_order),
      bunch_size(bunch_size) { }
    
    virtual ~BunchHashedLM() { }

    virtual bool isDeterministic() const {
      return true;
    }
    
    virtual bool requireHistoryManager() const {
      return false;
    }

    unsigned int getBunchSize() { return bunch_size; }

    void setBunchSize(unsigned int bunch_size) {
      this->bunch_size = bunch_size;
    }
  };

}; // closes namespace

#endif // BUNCH_HASHED_LM_H
