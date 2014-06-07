#ifndef HISTORY_BASED_LM_H
#define HISTORY_BASED_LM_H

#include <stdint.h>
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
      WordType* context_words;
      unsigned int context_size = 0;
      WordType aux_key = trie->getParent(key);
      
      // Go backward to get context size
      while (aux_key != trie->rootNode()) {
        context_size++;
        aux_key = trie->getParent(aux_key);
      }

      // If context size is maximum, compute score
      // and key and return them using the result
      // vector. Else, do nothing.
      if (context_size == (this->model->ngramOrder() - 1)) {
        // Context words must be collected from current
        // key, which shifts context to the left
        context_words = new WordType[context_size];
        aux_key = key;

        // Context words are collected
        for (int pos = context_size - 1; pos >= 0; --pos) {
          context_words[pos] = trie->getWord(aux_key);
          aux_key = trie->getParent(aux_key);
        }

        Score aux_score = privateGet(key, word, context_words, context_size);
        
        // Destination key is obtained traversing the trie
        aux_key = trie->rootNode();
        for (int i = 0; i < context_size; ++i)
          aux_key = trie->getChild(aux_key, context_words[i]);
        aux_key = trie->getChild(aux_key, word);

        // Append to the result vector
        result.push_back(KeyScoreBurdenTuple(aux_key,
                                             aux_score,
                                             burden));
      }
    }

    virtual void getNextKeys(const Key &key, WordType word,
                             vector<Key> &result) {
      WordType aux_key;

      aux_key = trie->getChild(key, word);
      result.push_back(aux_key);
    }

    virtual bool getZeroKey(Key &k) const {
      k = trie->rootNode();
      return true;
    }

    virtual void getInitialKey(Key &k) const {
      HistoryBasedLM<Key,Score> *mdl = static_cast<HistoryBasedLM<Key,Score>* >(this->model);
      WordType init_word = mdl->getInitWord();
      int context_length = this->model->ngramOrder() - 1;

      WordType aux = trie->rootNode();

      for (unsigned int i = 0; i < context_length; i++)
        aux = trie->getChild(aux, init_word);
      k = aux;
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
