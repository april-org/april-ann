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
#include "LM_interface.h"
#include "logbase.h"
#include "open_addressing_hash.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {
  
  using april_utils::vector;
  using april_utils::open_addr_hash;
  
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
      KeyScoreMultipleBurdenTuple(Key k, Score s, vector<Burden> burden_vector) :
        key_score(k,s), burden_vector(burden_vector) {}
    };

    ~BunchHashedLMInterface() {
    }
  protected:
    typedef open_addr_hash<WordType, vector<KeyScoreMultipleBurdenTuple> > WordResultHash;
    typedef open_addr_hash<Key, WordResultHash> KeyWordHash;

  private:
    KeyWordHash context_key_hash;

    vector<KeyScoreTuple> &bunchGet(Key, WordResultHash) {
      vector<KeyScoreTuple> bunch_result;
      return bunch_result;
    }

  public:
    virtual void get(const Key &key, WordType word,
                     Burden burden,
                     vector<KeyScoreBurdenTuple> &result,
                     Score threshold) {
      ;
    }

    virtual void getNextKeys(const Key &key, WordType word,
                             vector<Key> &result) {
      ;
    }

    virtual void clearQueries() {
      LMInterface<Key,Score>::clearQueries();
      context_key_hash.clear();
    }

    virtual void insertQuery(const Key &key, WordType word, Burden burden,
                             Score threshold) {
      // We hash the context key if it's not hashed
      if (not context_key_hash.search(key))
        context_key_hash[key] = WordResultHash();

      // We hash the word if it's not hashed
      if (not context_key_hash[key].search(word))
        context_key_hash[key][word] = KeyScoreMultipleBurdenTuple();

      // Add burden to the end of the burden vector
      context_key_hash[key][word].burden_vector.push_back(burden);
    }

    virtual const vector<KeyScoreBurdenTuple> &getQueries() const {
      // For each context key entry
      for (typename KeyWordHash::iterator it = context_key_hash.begin();
        it != context_key_hash.end(); ++it) {
        vector<KeyScoreTuple> bunch_result;
        unsigned int cur_result = 0;
        // We launch a bunchGet
        Key context_key = (*it)->first;
        WordResultHash word_hash = (*it)->second;

        bunch_result = bunchGet(context_key, word_hash);

        // Fill result vector with values from bunch_result
        for (typename WordResultHash::iterator it2 = word_hash.begin();
          it2 != word_hash.end(); ++it2) {
          WordType word = (*it)->first;
          KeyScoreMultipleBurdenTuple result_tuple = (*it2)->second;

          for (unsigned int i = 0; i < result_tuple.burden_vector.size(); ++i)
            this->result.push_back(KeyScoreBurdenTuple(bunch_result[cur_result].key,
                                                       bunch_result[cur_result].score,
                                                       result_tuple.burden_vector[i]));
          ++cur_result;
        }
      }
      return this->result;
    }

    virtual bool getZeroKey(Key &k) const {
      return true;
    }

    virtual void getInitialKey(Key &k) const {
      ;
    }

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

    virtual bool isDeterministic() const {
      return true;
    }
    
    virtual int ngramOrder() const {
      return ngram_order;
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
