#ifndef LANGUAGE_MODEL_INTERFACE
#define LANGUAGE_MODEL_INTERFACE

#include <stdint.h>
#include "logbase.h"
#include "referenced.h"
#include "trie_vector.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {
  
  using april_utils::vector;
  typedef uint32_t WordType;

  class LMHistoryManager;

  // Score is usually log_float
  // Key is usually uint32_t
  template <typename Key, typename Score>
  class LMModel; // forward declaration
  template <typename Key, typename Score>
  class HistoryBasedLM;
  template <typename Key, typename Score>
  class BunchHashedLM;
  
  // Score is usually log_float
  // Key is usually uint32_t
  
  /// LMInterface is the non-thread-safe wrapper around a LMModel.  The
  /// LMModel contains the 'static' (and usually thread-safe) data
  /// (e.g. the actual automata, the MLPs of a NNLM, etc.) whereas the
  /// LMInterface usually contains other data structures to perform the
  /// actual computation of LM queries. It also contains the
  /// LMHistoryManager
  template <typename Key, typename Score>
  class LMInterface : public Referenced {
    friend class LMModel<Key,Score>;
    
  public:
    
    /// This is a tuple with a LM key and an associated score
    struct KeyScoreTuple {
      Key key;
      Score score;
      KeyScoreTuple() {}
      KeyScoreTuple(Key k, Score s) :
        key(k), score(s) {}
    };
    
    /// This is the burden ignored by the LM but needed by our decoders
    struct Burden {
      int32_t id_key;
      int32_t id_word;
      Burden() {}
      Burden(int32_t id_key, int32_t id_word) :
        id_key(id_key), id_word(id_word) {}
      Burden(const Burden &other) :
      id_key(other.id_key), id_word(other.id_word) { }
    };
    
    /// This struct is the result produced by the LM query, a pair of
    /// (key,score), which is the next LM key and its associated score, and the
    /// burden struct, which currently contains two identifiers
    struct KeyScoreBurdenTuple {
      KeyScoreTuple key_score;
      Burden  burden;
      KeyScoreBurdenTuple() {}
      KeyScoreBurdenTuple(Key k, Score s, Burden burden) :
        key_score(k,s), burden(burden) {}
    };
    
    /// This struct is used for the insertQuery method which receives several
    /// words together
    struct WordIdScoreTuple {
      WordType word;
      int32_t  id_word;
      Score    score;
      WordIdScoreTuple() {}
      WordIdScoreTuple(WordType w, int32_t idw, Score s) :
        word(w), id_word(idw), score(s) {}
    };
    
  private:
    /// auxiliary result vector
    vector<KeyScoreBurdenTuple> result;

  protected:
    
    LMModel<Key,Score>* model;
    LMInterface(LMModel<Key,Score>* model) : model(model) {
      IncRef(model);
    }
    
  public:

    virtual ~LMInterface() { 
      DecRef(model);
    }

    /// retrieves the LMModel where this LMInterface was obtained
    virtual LMModel<Key, Score>* getLMModel() {
      return model;
    }
    
    // -------------- individual LM queries -------------

    // get is the most basic LM query method, receives the Key and the Word,
    // returns 0,1 or several results by pushing back to the vector result (the
    // vector is not cleared, and it is not needed to be cleared as
    // pre-condition)
    
    
    /// this method is the same get with an interface more similar to
    /// the bunch (multiple queries) mode
    /// TODO: threshold should have a default value
    virtual void get(const Key &key, WordType word, Burden burden,
                     vector<KeyScoreBurdenTuple> &result,
                     Score threshold) = 0;
    
    /// this method computes the next keys given a pair (key,word). It could be a
    /// non-deterministic LM. By default, it uses the standard get() method and
    /// discards the Burden and Score.
    virtual void getNextKeys(const Key &key, WordType word,
                             vector<Key> &result) {
      vector<KeyScoreBurdenTuple> aux_result;
      get(key, word, Burden(-1,-1), aux_result, Score::zero());
      for (typename vector<KeyScoreBurdenTuple>::iterator it = aux_result.begin();
           it != aux_result.end(); ++it)
        result.push_back(it->key_score.key);
    }
    
    // -------------- BUNCH MODE -------------
    // Note: we can freely mix insertQuery and insertQueries, but
    // individual queries (gets) and bunch mode queries should not be
    // mixed.

    /// the bunch mode operates in a state fashion, the first operation
    /// is clearQueries to reset the internal structures and the result
    /// vector
    void clearQueries() {
      result.clear();
    }

    /// call this method for each individual query.
    /// The default implementation use the get method
    virtual void insertQuery(const Key &key, WordType word, Burden burden,
                             Score threshold) {
      get(key,word,burden,result,threshold);
    }
    
    /// this method can be naively converted into a series of
    /// insertQuery calls, but it can be optimized for some 
    virtual void insertQueries(const Key &key, int32_t id_key,
                               vector<WordIdScoreTuple> words,
                               bool is_sorted=false) {
      UNUSED_VARIABLE(is_sorted);
      // default behavior
      for (typename vector<WordIdScoreTuple>::iterator it = words.begin();
	   it != words.end(); ++it)
        insertQuery(key, it->word, Burden(id_key, it->id_word), it->score);
    }

    /// this method may perform the 'actual' computation of LM queries
    /// in some implementations which can benefit of bunch mode (for
    /// instance, NNLMs may profit this feature for grouping all
    /// queries associated to the same Key, and can perform serveral
    /// forward steps in bunch mode). Other LMs such as those based on
    /// automata (e.g. ngram_lira) may perform the LM queries in the
    /// insert method so that here they only have to return the result
    /// vector.
    virtual const vector<KeyScoreBurdenTuple> &getQueries() const {
      return result;
    }

    /// an upper bound on the best transition probability, usually
    /// pre-computed in the model
    virtual Score getBestProb() const = 0;

    /// an upper bound on the best transition probability departing
    /// from key, usually pre-computed in the model
    virtual Score getBestProb(const Key &k) const = 0;

    /// initial key is the initial context cue
    virtual void getInitialKey(Key &k) const = 0;

    /// this method returns false and does nothing on LMs without zero key
    virtual bool getZeroKey(Key &k) const {
      UNUSED_VARIABLE(k);
      // default behavior
      return false;
    }

    // I don't like this method, it seems to assume that there is only one final state
    //virtual void getFinalKey(Key &k) const = 0;
    // replaced by getFinalScore method

    // returns the score associated to the probability of being a
    // final state. This method is not const since it may depend on
    // get which is not const either. An score is provided as pruning
    // technique (in the same way as with get method):
    virtual Score getFinalScore(const Key &k, Score threshold) = 0;
  };

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
  class BunchHashedLMInterface : public LMInterface <Key,Score> {
    friend class BunchHashedLM<Key,Score>;
  private:

  protected:
    
    BunchHashedLMInterface(BunchHashedLM<Key,Score>* model) :
      LMInterface<Key,Score>(model) {
    }
    
  public:

    ~BunchHashedLMInterface() {
    }

    virtual void get(const Key &key, WordType word,
                     typename LMInterface<Key,Score>::Burden burden,
                     vector<typename LMInterface<Key,Score>::KeyScoreBurdenTuple> &result,
                     Score threshold) {
      ;
    }

    virtual void getNextKeys(const Key &key, WordType word,
                             vector<Key> &result) {
      ;
    }

    virtual bool getZeroKey(Key &k) const {
      return true;
    }

    virtual void getInitialKey(Key &k) const {
      ;
    }

  };

  /// The LMModel is the thread-safe part of the LM, where the model data is
  /// stored
  template <typename Key, typename Score>
  class LMModel : public Referenced {
  public:

    LMModel() : Referenced() {}

    virtual bool isDeterministic() const {
      // default behavior
      return false;
    }
    
    // FIXME: Is it needed?
    // return -1 when it is not an ngram
    virtual int ngramOrder() const {
      // default behavior
      return -1;
    }
    

    virtual bool requireHistoryManager() const {
      // default behavior
      return true;
    }

    virtual LMInterface<Key,Score>* getInterface() = 0;
  };

  class LMHistoryManager : public Referenced {
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

  typedef LMInterface<uint32_t,log_float> LMInterfaceUInt32LogFloat;
  typedef LMModel<uint32_t,log_float> LMModelUInt32LogFloat;

  typedef HistoryBasedLMInterface<uint32_t, log_float> HistoryBasedLMInterfaceUInt32LogFloat;
  typedef HistoryBasedLM<uint32_t, log_float> HistoryBasedLMUInt32LogFloat;
}; // closes namespace

#endif // LANGUAGE_MODEL_INTERFACE
