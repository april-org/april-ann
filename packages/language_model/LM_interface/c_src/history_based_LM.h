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
#ifndef HISTORY_BASED_LM_H
#define HISTORY_BASED_LM_H

#include <stdint.h>
#include "error_print.h"
#include "LM_interface.h"
#include "logbase.h"
#include "trie_vector.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {
  
  using april_utils::vector;
  
  template <typename Key, typename Score>
  class HistoryBasedLM;
  
  /// HistoryBasedLMInterface documentation ...
  template <typename Key, typename Score>
  class HistoryBasedLMInterface : public LMInterface <Key,Score> {
    friend class HistoryBasedLM<Key,Score>;
  private:

    virtual Score privateGet(const Key &key,
                             WordType word,
                             WordType *context_words,
                             unsigned int context_size) = 0;

    april_utils::TrieVector *trie;

  protected:
    
    HistoryBasedLMInterface(HistoryBasedLM<Key,Score>* model) :
      LMInterface<Key,Score>(model) {
      trie = model->getTrieVector();
      IncRef(trie);
    }

    virtual Score privateGetFinalScore(const Key &key,
                                       WordType *context_words,
                                       unsigned int context_size) = 0;

    unsigned int getContextProperties(const Key &key,
                                      WordType *context_words) {
      Key aux_key = key;

      // Go backward to get context size and context words. Context words must
      // be collected from current key, which shifts context to the left
      int pos = this->model->ngramOrder() - 1;
      while (aux_key != trie->rootNode() && pos > 0) {
        context_words[pos-1] = trie->getWord(aux_key);
        aux_key = trie->getParent(aux_key);
        --pos;
      }
      if (aux_key != trie->rootNode())
        ERROR_EXIT(256, "Overflow filling context words from current key\n");
      // compute context_size from position in context_words
      return (this->model->ngramOrder() - 1 - pos);                               
    }

    typedef typename LMInterface<Key,Score>::KeyScoreBurdenTuple KeyScoreBurdenTuple;
    typedef typename LMInterface<Key,Score>::Burden Burden;

  public:

    virtual ~HistoryBasedLMInterface() {
      DecRef(trie);
    }

    virtual void get(const Key &key, WordType word,
                     Burden burden,
                     vector<KeyScoreBurdenTuple> &result,
                     Score threshold) {
      UNUSED_VARIABLE(threshold);
      WordType* context_words = new WordType[this->model->ngramOrder() - 1];
      const unsigned int context_size = getContextProperties(key,
                                                             context_words);
      // If context size is maximum, compute score
      // and key and return them using the result
      // vector. Else, do nothing.
      if (context_size == (this->model->ngramOrder() - 1)) {
        Score aux_score = privateGet(key, word, context_words, context_size);
        // Destination key is obtained traversing the trie starting from
        // context_word[1]
        Key aux_key = trie->rootNode();
        for (int i = 1; i < context_size; ++i)
          aux_key = trie->getChild(aux_key, context_words[i]);
        aux_key = trie->getChild(aux_key, word);

        // Append to the result vector
        result.push_back(KeyScoreBurdenTuple(aux_key,
                                             aux_score,
                                             burden));
      }
      delete[] context_words;
    }

    virtual void getNextKeys(const Key &key, WordType word,
                             vector<Key> &result) {
      Key aux_key;
      WordType* context_words = new WordType[this->model->ngramOrder() - 1];
      const unsigned int context_size = getContextProperties(key,
                                                             context_words);
      // If context is maximum
      if (context_size == (this->model->ngramOrder() - 1)) {
        // Destination key is obtained traversing the trie starting from
        // context_word[1]
        aux_key = trie->rootNode();
        for (int i = 1; i < context_size; ++i)
          aux_key = trie->getChild(aux_key, context_words[i]);
        aux_key = trie->getChild(aux_key, word);
      } else {
        // Else destination key is obtained from given key and word 
        aux_key = trie->getChild(key, word);
      }
      // Append to the result vector
      result.push_back(aux_key);
      delete[] context_words;
    }

    virtual bool getZeroKey(Key &k) const {
      k = trie->rootNode();
      return true;
    }

    virtual void getInitialKey(Key &k) const {
      HistoryBasedLM<Key,Score> *mdl = static_cast<HistoryBasedLM<Key,Score>* >(this->model);
      WordType init_word = mdl->getInitWord();
      int context_length = this->model->ngramOrder() - 1;

      k = trie->rootNode();

      for (unsigned int i = 0; i < context_length; i++)
        k = trie->getChild(k, init_word);
    }

    virtual Score getFinalScore(const Key &k, Score threshold) {
      UNUSED_VARIABLE(threshold);
      WordType* context_words = new WordType[this->model->ngramOrder() - 1];
      const unsigned int context_size = getContextProperties(k, context_words);

      // If context size is maximum, compute score
      // and key and return them using the result
      // vector. Else, do nothing.
      // TODO: Fix this logic.
      if (context_size == (this->model->ngramOrder() - 1)) {
        Score score = privateGetFinalScore(k, context_words, context_size);
        delete[] context_words;
        return score;
      } else return Score::zero();
    }
  };
  
  template <typename Key, typename Score>
  class HistoryBasedLM : public LMModel <Key,Score> {
  private:
    int ngram_order;
    WordType init_word;
    april_utils::TrieVector *trie_vector;

  public:

    HistoryBasedLM(int ngram_order,
                   WordType init_word,
                   april_utils::TrieVector *trie_vector) : 
      LMModel<Key,Score>(),
      ngram_order(ngram_order),             
      init_word(init_word),             
      trie_vector(trie_vector) {
      IncRef(trie_vector);
    }

    ~HistoryBasedLM() {
      DecRef(trie_vector);
    }

    virtual bool isDeterministic() const {
      return true;
    }
    
    virtual int ngramOrder() const {
      return ngram_order;
    }

    virtual bool requireHistoryManager() const {
      return true;
    }
    
    WordType getInitWord() {
      return init_word;
    }

    april_utils::TrieVector* getTrieVector() {
      return trie_vector;
    }

    // this class is also abstract and hence does not implement
    // getInterface
    // virtual LMInterface<Key,Score>* getInterface() = 0;
  };
  
  typedef HistoryBasedLMInterface<uint32_t, log_float> HistoryBasedLMInterfaceUInt32LogFloat;
  typedef HistoryBasedLM<uint32_t, log_float> HistoryBasedLMUInt32LogFloat;
}; // closes namespace

#endif // HISTORY_BASED_LM_H
