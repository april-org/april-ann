#ifndef LANGUAGE_MODEL_INTERFACE
#define LANGUAGE_MODEL_INTERFACE

#include <stdint.h>
#include "logbase.h"
#include "referenced.h"
#include "vector.h"

namespace language_models {
  
  using april_utils::vector;
  typedef int32_t WordType;

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
    
    struct KeyScoreTuple {
      Key key;
      Score score;
      KeyScoreTuple() {}
      KeyScoreTuple(Key k, Score s) :
	key(k), score(s) {}
    };
    
    struct KeyScoreIdTuple {
      Key     key;
      Score   score;
      int32_t idKey;
      int32_t idWord;
      KeyScoreIdTuple() {}
      KeyScoreIdTuple(Key k, Score s, int32_t ik, int32_t iw) :
	key(k), score(s), idKey(ik), idWord(iw) {}
    };

    struct WordIdScoreTuple {
      WordType word;
      int32_t  idWord;
      Score    score;
      WordIdScoreTuple() {}
      WordIdScoreTuple(WordType w, int32_t idw, Score s) :
	word(w), idWord(idw), score(s) {}
    };
    
    vector<KeyScoreIdTuple> result;
    
    virtual ~LMInterface() { }

    // retrieves the LMModel where this LMInterface was obtained
    virtual LMModel* getLMModel() = 0;
    
    // -------------- individual LM queries -------------

    // get is the most basic LM query method, receives the Key and the Word,
    // returns 0,1 or several results by overwritting the vector result
    // which is cleared in the method (not need to be cleared before)
    

    // this method is the same get with an interface more similar to
    // the bunch (multiple queries) mode
    // returns the size of result
    // TODO: threshold should have a default value
    virtual int get(const Key &key, int32_t idKey,
		     WordType word, int32_t idWord,
		     vector<KeyScoreIdTuple> &result, Score threshold) = 0;

    // returns the size of result
    virtual int get(const Key &key, WordType word,
		     vector<KeyScoreTuple> &result, Score threshold) {
      // default behavior 
      resul.clear();
      vector<KeyScoreIdTuple> aux;
      get(key,0,aux,threshold);
      for (vector<KeyScoreIdTuple>::iterator it = aux.begin();
	   it != aux.end(); ++it)
	resul.push_back(KeyScoreTuple(it->key,it->score));
      return resul.size();
    }

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
			     Score threshold) {
      // default implementation use the method get
      vector<KeyScoreIdTuple> aux;
      get(key,idKey,word,idWord,aux,threshold);
      for (vector<KeyScoreIdTuple>::iterator it = aux.begin();
	   it != aux.end(); ++it)
	resul.push_back(*it);
    }
    
    // this method can be naively converted into a series of
    // insertQuery calls, but it can be optimized for some 
    virtual void insertQueries(const Key &key, int32_t idKey,
			       vector<WordIdScoreTuple> words) {
      // default behavior
      for (vector<WordIdScoreTuple>::iterator it = words.begin();
	   it != words.end(); ++it)
	insertQuery(key, idKey, it->word, it->idWord, it->score);
    }

    // this method may perform the 'actual' computation of LM queries
    // in some implementations which can benefit of bunch mode (for
    // instance, NNLMs may profit this feature for grouping all
    // queries associated to the same Key, and can perform serveral
    // forward steps in bunch mode). Other LMs such as those based on
    // automata (e.g. ngram_lira) may perform the LM queries in the
    // insert method so that here they only have to return the result
    // vector.
    // returns result size
    virtual int getQueries(vector<KeyScoreIdentifiersTuple> &result) {
      // default behavior
      result = this->result;
      return result.size();
    }

    // an upper bound on the best transition probability, usually
    // pre-computed in the model
    virtual Score getBestProb() = 0;

    // an upper bound on the best transition probability departing
    // from key, usually pre-computed in the model
    virtual Score getBestProb(Key &k) = 0;

    // initial word is the initial context cue
    virtual void getInitialKey(Word initial_word, Key &k) = 0;

    // this method returns false and does nothing on non-ngram LMs
    virtual bool getZerogramKey(Key &k) {
      // default behavior
      return false;
    }

    // I don't like this method, it seems to assume that there is only one final state
    virtual void  getFinalKey(Word final_word, Key &k) = 0;
    // replace by this method?
    virtual Score getFinalScore(Word final_word, Key &k) = 0;
  };

  template <typename Key, typename Score>
  class LMModel : public Referenced {
  public:

    virtual bool isDeterministic() {
      // default behavior
      return false;
    }

    // return -1 when it is not an ngram
    virtual int ngramOrder() {
      // default behavior
      return -1;
    }

    virtual bool requireHistoryManager() {
      // default behavior
      return true;
    }

    // TODO: LMHistoryManager does not yet exist, it is essentially a
    // (wrapper to a) TrieVector
    virtual LMInterface<Key,Score>* getInterface(LMHistoryManager *hmanager=0) = 0;

  };

}; // closes namespace

#endif // LANGUAGE_MODEL_INTERFACE
