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
#include "april_assert.h"
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
    unsigned int getContextProperties(Key key,
                                      WordType *context_words,
                                      unsigned int &offset) {
      // Go backward to get context size and context words. Context words must
      // be collected from current key, which shifts context to the left
      unsigned int pos = this->model->ngramOrder() - 1;
      while (key != trie->rootNode() && pos > 0) {
        context_words[pos-1] = trie->getWord(key);
        key = trie->getParent(key);
        --pos;
      }
      if (key != trie->rootNode())
        ERROR_EXIT(256, "Overflow filling context words from current key\n");
      // return offset in context_words
      offset = pos;
      // compute context_size from position in context_words
      return (this->model->ngramOrder() - 1 - pos);
    }
    
    Key getDestinationKey(const WordType *context_words,
                          const unsigned int offset,
                          const unsigned int context_size,
                          const WordType word) {
      // in case the offset==0 (context_size == order-1), we need to swipe out
      // one word
      int begin = (offset == 0) ? 1 : offset, end = offset + context_size;
      // Destination key is obtained traversing context_words from begin to
      // end-1
      Key dest_key = trie->rootNode();
      for (int i = begin; i < end; ++i)
        dest_key = trie->getChild(dest_key, context_words[i]);
      // last transition uses the given word
      dest_key = trie->getChild(dest_key, word);
      return dest_key;
    }

    april_utils::TrieVector *trie;
    WordType *context_words;

  protected:

    HistoryBasedLMInterface(HistoryBasedLM<Key,Score>* model) :
      LMInterface<Key,Score>(model) {
      trie = model->getTrieVector();
      IncRef(trie);
      // we put one more word to allow storing next word at end of this array
      context_words = new WordType[model->ngramOrder()];
    }

    /// returns true in case the probability could be computed
    virtual bool privateGet(Key key,
                            WordType word,
                            const WordType *context_words,
                            unsigned int context_size,
                            Score threshold,
                            Score &score) = 0;

    /// returns true in case the probability could be computed
    virtual bool privateGetFinalScore(Key key,
                                      const WordType *context_words,
                                      unsigned int context_size,
                                      Score threshold,
                                      Score &score) = 0;

    /// returns true in case the probability could be computed
    virtual bool privateBestProb(Key key,
                                 const WordType *context_words,
                                 unsigned int context_size,
                                 Score &score) = 0;

    typedef typename LMInterface<Key,Score>::KeyScoreBurdenTuple KeyScoreBurdenTuple;
    typedef typename LMInterface<Key,Score>::Burden Burden;

  public:

    virtual ~HistoryBasedLMInterface() {
      DecRef(trie);
      delete[] context_words;
    }

    virtual void get(Key key, WordType word,
                     Burden burden,
                     vector<KeyScoreBurdenTuple> &result,
                     Score threshold) {
      const int order = this->model->ngramOrder();
      unsigned int offset;
      const unsigned int context_size = getContextProperties(key,
                                                             context_words,
                                                             offset);
      // Compute score with the retrieved context. In case of success, return
      // them using the result vector. Else, do nothing.
      Score score;
      if (privateGet(key, word, context_words + offset, context_size,
                     threshold, score)) {
        // Destination key is obtained traversing the trie
        Key dest_key = getDestinationKey(context_words, offset,
                                         context_size, word);
        // Append to the result vector
        result.push_back(KeyScoreBurdenTuple(dest_key, score, burden));
      }
    }

    virtual void getNextKeys(Key key, WordType word,
                             vector<Key> &result) {
      unsigned int offset;
      const unsigned int context_size = getContextProperties(key,
                                                             context_words,
                                                             offset);
      Key dest_key = getDestinationKey(context_words, offset,
                                       context_size, word);
      // Append to the result vector
      result.push_back(dest_key);
    }

    virtual bool getZeroKey(Key &k) const {
      k = trie->rootNode();
      return true;
    }

    virtual Key getInitialKey() {
      HistoryBasedLM<Key,Score> *mdl;
      mdl = static_cast<HistoryBasedLM<Key,Score>* >(this->model);
      WordType init_word = mdl->getInitWord();
      int context_length = this->model->ngramOrder() - 1;
      Key k = trie->rootNode();
      for (unsigned int i = 0; i < context_length; i++)
        k = trie->getChild(k, init_word);
      return k;
    }

    virtual Score getFinalScore(Key k, Score threshold) {
      unsigned int offset;
      const unsigned int context_size = getContextProperties(k, context_words,
                                                             offset);
      // Compute score with the retrieved context. In case of fail, return
      // zero probability
      Score score;
      if (!privateGetFinalScore(k, context_words+offset, context_size,
                                threshold, score))
        score = Score::zero();
      return score;
    }

    Score getBestProb(Key k) {
      unsigned int offset;
      const unsigned int context_size = getContextProperties(k, context_words,
                                                             offset);
      // Compute score with the retrieved context. In case of fail, return
      // zero probability
      Score score;
      if (!privateBestProb(k, context_words+offset, context_size, score))
        score = Score::zero();
      return score;
    }

    // virtual Score getBestProb() const = 0;
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
      if (ngram_order <= 0)
        ERROR_EXIT(128, "Impossible to build HistoryBasedLM with <= 0 order\n");
    }

    virtual ~HistoryBasedLM() {
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
