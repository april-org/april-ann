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
#ifndef LANGUAGE_MODEL_INTERFACE_H
#define LANGUAGE_MODEL_INTERFACE_H

#include <stdint.h>
#include "logbase.h"
#include "referenced.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {
  
  /**
   * @brief The standard WordType is an unsigned integer of 32 bits.
   *
   * @note 4G words are possible to be represented.
   */
  typedef uint32_t WordType;
  
  // forward declaration
  class LMHistoryManager;

  // Score is usually log_float
  // Key is usually uint32_t
  template <typename Key, typename Score>
  class LMModel; // forward declaration
  
  /**
   * @brief LMInterface is the non-thread-safe warpper around a LMModel.
   *
   * The LMModel contains the 'static' (and usually thread-safe) data (e.g. the
   * actual automata, the MLPs of a NNLM, etc.) where as the LMInterface usually
   * contains other data structures to perform the actual computation of LM
   * queries.
   *
   * @note In history based LMs this class contains a history manager, usually a
   * april_utils::TrieVector instance.
   *
   * @note Score is usually april_utils::log_float.
   *
   * @note Key is usually WordType (uint32_t).
   */
  template <typename Key, typename Score>
  class LMInterface : public Referenced {
    /// The LMModel is a friend class because LMInterface constructor is
    /// protected, and it is forced to be called from LMModel::getInterface()
    /// method.
    friend class LMModel<Key,Score>;
    
  public:
    
    /// This is a tuple with a LM Ley and an associated Score.
    struct KeyScoreTuple {
      Key key;
      Score score;
      KeyScoreTuple() {}
      KeyScoreTuple(Key k, Score s) :
        key(k), score(s) {}
    };
    
    /// This is the burden ignored by the LM but needed by our decoders.
    struct Burden {
      int32_t id_key;  ///< Usually a number related with the context.
      int32_t id_word; ///< Usually a number related with the next word.
      Burden() {}
      Burden(int32_t id_key, int32_t id_word) :
        id_key(id_key), id_word(id_word) {}
      Burden(const Burden &other) :
      id_key(other.id_key), id_word(other.id_word) { }
    };
    
    /**
     * @brief This struct is the result of a LM query.
     *
     * This struct is the result produced by the LM query, a pair of
     * (Key,S), which is the next LM Key and its associated S, and the
     * Burden struct, which currently contains two identifiers.
     */
    struct KeyScoreBurdenTuple {
      KeyScoreTuple key_score; ///< The (Key,Score) pair.
      Burden  burden;          ///< The Burden related with this pair.
      KeyScoreBurdenTuple() {}
      KeyScoreBurdenTuple(Key k, Score s, Burden burden) :
        key_score(k,s), burden(burden) {}
    };
    
    /// This struct is used for the insertQuery method which receives several
    /// words together.
    struct WordIdScoreTuple {
      WordType word;
      int32_t  id_word;
      Score    score;
      WordIdScoreTuple() {}
      WordIdScoreTuple(WordType w, int32_t idw, Score s) :
        word(w), id_word(idw), score(s) {}
    };
    
  protected:
    /// Auxiliary result vector.
    april_utils::vector<KeyScoreBurdenTuple> result;
    
    /// The model reference.
    LMModel<Key,Score>* model;
    
    /// The protected constructor, only callable from LMModel::getInterface()
    /// method.
    LMInterface(LMModel<Key,Score>* model) : model(model) {
      IncRef(model);
    }
    
  public:
    
    /// Destructor.
    virtual ~LMInterface() { 
      DecRef(model);
    }

    /// Retrieves the LMModel where this LMInterface was obtained.
    virtual LMModel<Key, Score>* getLMModel() {
      return model;
    }
    
    // -------------- individual LM queries -------------

    // get is the most basic LM query method, receives the Key and the Word,
    // returns 0,1 or several results by pushing back to the vector result (the
    // vector is not cleared, and it is not needed to be cleared as
    // pre-condition)
    
    
    /**
     * @brief Queries the model with a (Key,WordType,Burden) tuple.
     *
     * This method returns all the (Key,WordType) pairs of a transition in the
     * LMModel by using the state @c key with the given @c word
     * instance. Additionally a @c threshold is given to perform beam pruning.
     *
     * @param key - The context or state from where the transition starts.
     *
     * @param word - The transition word.
     *
     * @param[in,out] result - An april_utils::vector with all the destination
     * states.
     *
     * @param threshold - The threshold for the beam pruning.
     *
     * @note The result vector is not cleared, so all the resulting states will
     * be appended using april_utils::vector::push_back() method.
     */
    virtual void get(Key key, WordType word, Burden burden,
                     april_utils::vector<KeyScoreBurdenTuple> &result,
                     Score threshold) = 0;
    
    /// this method computes the next keys given a pair (key,word). It could be a
    /// non-deterministic LM. By default, it uses the standard get() method and
    /// discards the Burden and Score.
    virtual void getNextKeys(Key key, WordType word,
                             april_utils::vector<Key> &result) {
      april_utils::vector<KeyScoreBurdenTuple> aux_result;
      get(key, word, Burden(-1,-1), aux_result, Score::zero());
      for (typename april_utils::vector<KeyScoreBurdenTuple>::iterator it = aux_result.begin();
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
    virtual void clearQueries() {
      result.clear();
    }

    /// call this method for each individual query.
    /// The default implementation use the get method
    virtual void insertQuery(Key key, WordType word, Burden burden,
                             Score threshold) {
      get(key,word,burden,result,threshold);
    }
    
    /// this method can be naively converted into a series of
    /// insertQuery calls, but it can be optimized for some 
    virtual void insertQueries(Key key, int32_t id_key,
                               april_utils::vector<WordIdScoreTuple> words,
                               bool is_sorted=false) {
      UNUSED_VARIABLE(is_sorted);
      // default behavior
      for (typename april_utils::vector<WordIdScoreTuple>::iterator it = words.begin();
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
    virtual const april_utils::vector<KeyScoreBurdenTuple> &getQueries() {
      return result;
    }

    /// an upper bound on the best transition probability, usually
    /// pre-computed in the model
    virtual Score getBestProb() const = 0;

    /// an upper bound on the best transition probability departing
    /// from key, usually pre-computed in the model
    virtual Score getBestProb(Key k) = 0;

    /// initial key is the initial state (initial context cue)
    virtual Key getInitialKey() = 0;

    /// this method returns false and does nothing on LMs without zero key
    virtual bool getZeroKey(Key &k) const {
      UNUSED_VARIABLE(k);
      // default behavior
      return false;
    }

    // returns the score associated to the probability of being a
    // final state. This method is not const since it may depend on
    // get which is not const either. An score is provided as pruning
    // technique (in the same way as with get method):
    virtual Score getFinalScore(Key k, Score threshold) = 0;
  };

  /**
   * @brief The LMModel is the thread-safe part of the LM.
   *
   * This class is where the model data is stored, it is thread-safe and
   * therefore can be shared between multiples threads.
   *
   * @note Score is usually a april_utils::log_float.
   *
   * @note Key is usually a WordType (uint32_t).
   */
  template <typename Key, typename Score>
  class LMModel : public Referenced {
  public:
    
    /// Constructor
    LMModel() : Referenced() {}
    /// Destructor
    virtual ~LMModel() {}
    
    /**
     * @brief Indicates if the model is deterministic or not.
     *
     * @note the default implementation is <tt>return false</tt>.
     */
    virtual bool isDeterministic() const {
      // default behavior
      return false;
    }
    
    /**
     * @brief Indicates the order of the N-gram.
     *
     * @note By default it returns @c -1 indicating that the model is not an
     * N-gram model.
     */
    virtual int ngramOrder() const {
      // default behavior
      return -1;
    }
    
    /**
     * @brief Indicates if the model requires a history manager.
     *
     * @note By default it is <tt>return true</tt>
     */
    virtual bool requireHistoryManager() const {
      // default behavior
      return true;
    }
    
    /**
     * @brief Returns an instance of the interface with the model.
     *
     * All derived classes must implement this getInterface() method which must
     * return the proper instance of a LMInterface derived class corresponding
     * with a proper LMModel. Every thread which needs to query the language
     * model needs to get an interface using this method.
     */
    virtual LMInterface<Key,Score>* getInterface() = 0;
  };
  
  typedef LMInterface<uint32_t,april_utils::log_float> LMInterfaceUInt32LogFloat;
  typedef LMModel<uint32_t,april_utils::log_float> LMModelUInt32LogFloat;

}; // closes namespace

#endif // LANGUAGE_MODEL_INTERFACE_H
