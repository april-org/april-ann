#ifndef LANGUAGE_MODEL_INTERFACE
#define LANGUAGE_MODEL_INTERFACE

#include <stdint.h>
#include "logbase.h"
#include "referenced.h"
#include "vector.h"

namespace language_models {
  
  using april_utils::vector;
  typedef int32_t Word;

  // Score is usually log_float
  // Key is usually uint32_t
  template <typename Key, typename Score>
  class LMModel; // forward declaration
  
  // Score is usually log_float
  // Key is usually uint32_t
  template <typename Key, typename Score>
  // LMInterface is the non-thread-safe wrapper around a LMModel.  The
  // LMModel contains the 'static' (and usually thread-safe) data
  // (e.g. the actual automata, the MLPs of a NNLM, etc.) whereas the
  // LMInterface usually contains other data structures to perform the
  // actual computation of LM queries. It also contains the
  // LMHistoryManager
  class LMInterface : public Referenced {
    
  public:
    
    vector<KeyScoreIdTuple> result;
    
    struct KeyScoreTuple {
      Key key;
      Score score;
    };
    
    struct KeyScoreIdTuple {
      Key     key;
      Score   score;
      int32_t idKey;
      int32_t idWord;
    };

    struct WordIdScoreTuple {
      Word    word
      int32_t idWord;
      Score   score;
    };
    
    virtual ~LMInterface() { }

    // retrieves the LMModel where this LMInterface was obtained
    virtual LMModel* getLMModel() = 0;
    
    // -------------- individual LM queries -------------

    // get is the most basic LM query method, receives the Key and the Word,
    // returns 0,1 or several results by overwritting the vector result
    // which is cleared in the method (not need to be cleared before)
    
    // TODO: threshold should have a default value
    virtual void get(const Key &key, Word word, vector<KeyScoreTuple> &result, Score threshold) = 0;

    // this method is the same get with an interface more similar to
    // the bunch (multiple queries) mode
    virtual void get(const Key &key, int32_t idKey, Word word, int32_t idWord,
		     vector<KeyScoreIdTuple> &result, Score threshold) = 0;


    // -------------- BUNCH MODE -------------
    // Note: we can freely mix insertQuery and insertQueries, but
    // individual queries (gets) and bunch mode queries should not be
    // mixed.

    // the bunch mode operates in a state fashion, the first operation
    // is clearQueries to reset the internal structures and the result
    // vector
    void clearQueries() {
      result.clear();
    }

    // call this method for each individual query
    virtual void insertQuery(const Key &key, int32_t idKey,
			     Word word, int32_t idWord,
			     Score threshold);
    
    // this method can be naively converted into a series of
    // insertQuery calls, but it can be optimized for some 
    virtual void insertQueries(const Key &key, int32_t idKey,
			       vector<WordIdScoreTuple> words);

    // this method may perform the 'actual' computation of LM queries
    // in some implementations which can benefit of bunch mode (for
    // instance, NNLMs may profit this feature for grouping all
    // queries associated to the same Key, and can perform serveral
    // forward steps in bunch mode). Other LMs such as those based on
    // automata (e.g. ngram_lira) may perform the LM queries in the
    // insert method so that here they only have to return the result
    // vector.
    virtual void getQueries(vector<KeyScoreIdentifiersTuple> &result);


    // an upper bound on the best transition probability, usually
    // pre-computed in the model
    virtual Score getBestProb() = 0;

    // an upper bound on the best transition probability departing
    // from key, usually pre-computed in the model
    virtual Score getBestProb(Key &k) = 0;

    // initial word is the initial context cue
    virtual void  getInitialKey(Word initial_word, Key &k) = 0;

    // I don't like this method, it seems to assume that there is only one final state
    virtual void  getFinalKey(Word final_word, Key &k) = 0;
    // replace by this method?
    virtual Score getFinalScore(Word final_word, Key &k) = 0;
  };

  template <typename Key, typename Score>
  class LMModel : public Referenced {
  public:
    // return -1 when it is not an ngram
    int ngramOrder();

    bool requireHistoryManager();

    // TODO: LMHistoryManager does not yet exist, it is essentially a
    // (wrapper to a) TrieVector
    LMInterface<Key,Score>* getInterface(LMHistoryManager *hmanager=0);

  }:

}; // closes namespace

#endif // LANGUAGE_MODEL_INTERFACE
