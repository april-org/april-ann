#ifndef LANGUAGE_MODEL_INTERFACE
#define LANGUAGE_MODEL_INTERFACE

#include <stdint.h>
#include "logbase.h"
#include "referenced.h"
#include "vector.h"

namespace LanguageModels {
  
  using april_utils::vector;
  typedef int32_t WordType;

  class LMHistoryManager;

  // Score is usually log_float
  // Key is usually uint32_t
  template <typename Key, typename Score>
  class LMModel; // forward declaration
  
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
      Burden(const Burden &other) :
	idKey(other.id_key), idWord(other.id_word) { }
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
	word(w), idWord(idw), score(s) {}
    };
    
  private:
    
    /// auxiliary result vector
    vector<KeyScoreBurdenTuple> result;
    
  public:
    
    virtual ~LMInterface() { }

    /// retrieves the LMModel where this LMInterface was obtained
    virtual LMModel<Key, Score>* getLMModel() = 0;
    
    // -------------- individual LM queries -------------

    // get is the most basic LM query method, receives the Key and the Word,
    // returns 0,1 or several results by overwritting the vector result
    // which is cleared in the method (not need to be cleared before)
    
    
    /// this method is the same get with an interface more similar to
    /// the bunch (multiple queries) mode
    /// TODO: threshold should have a default value
    virtual void get(const Key &key, WordType word, Burden burden,
		     vector<KeyScoreBurdenTuple> &result, Score threshold) = 0;
    
    // returns the size of result
    virtual void get(const Key &key, WordType word,
		     vector<KeyScoreTuple> &result, Score threshold) {
      // default behavior 
      vector<KeyScoreBurdenTuple> aux;
      get(key,0,aux,threshold);
      for (typename vector<KeyScoreBurdenTuple>::iterator it = aux.begin();
	   it != aux.end(); ++it)
	result.push_back(KeyScoreTuple(it->key,it->score));
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

    /// this method returns false and does nothing on non-ngram LMs
    virtual bool getZeroGramKey(Key &k) const {
      // default behavior
      return false;
    }

    // I don't like this method, it seems to assume that there is only one final state
    virtual void  getFinalKey(Key &k) const = 0;
    // replace by this method?
    virtual Score getFinalScore(const Key &k) const = 0;
  };
  
  /// The LMModel is the thread-safe part of the LM, where the model data is
  /// stored
  template <typename Key, typename Score>
  class LMModel : public Referenced {
  public:

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

    // TODO: LMHistoryManager does not yet exist, it is essentially a
    // (wrapper to a) TrieVector
    virtual LMInterface<Key,Score>* getInterface() = 0;
  };

  class LMHistoryManager : public Referenced {
  };

  typedef LMInterface<uint32_t,log_float> LMInterfaceUInt32LogFloat
  typedef LMModel<uint32_t,log_float> LMModelUInt32LogFloat

}; // closes namespace

#endif // LANGUAGE_MODEL_INTERFACE
