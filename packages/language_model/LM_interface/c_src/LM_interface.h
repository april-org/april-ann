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
#include "smart_ptr.h"
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
   * Two modes are available to query the language models:
   *
   * - One-by-one: get() based methods query the language model and store its
   *   result into an AprilUtils::vector.
   *
   * - Bunch-mode: insertQuery(), getQueries(), clearQueries() methods allow to
   *   list a bunch of queries into a temporary data structure using
   *   insertQuery() method, the language model computes its output in the
   *   getQueries() method, and finally the clearQueries() method allows to
   *   reset the temporary list of queries. This mode uses the LMInterface
   *   instance to store the state of the computation.
   *
   * @note Language models can be deterministic or not, therefore, this
   * interface methods produce lists of (key,score) pairs with @c size()>=0.
   *
   * @note In history based LMs this class contains a history manager, usually a
   * AprilUtils::TrieVector instance.
   *
   * @note Score is usually AprilUtils::log_float.
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
    
    /// This struct is used for the insertQueries() method which receives several
    /// words together.
    struct WordIdScoreTuple {
      WordType word;
      int32_t  id_word; ///< A half part of the Burden (see insertQueries()).
      Score    score;
      WordIdScoreTuple() {}
      WordIdScoreTuple(WordType w, int32_t idw, Score s) :
        word(w), id_word(idw), score(s) {}
    };
    
    struct WordScoreTuple {
      WordType word;
      Score score;
      WordScoreTuple() {}
      WordScoreTuple(WordType w, Score s) :
        word(w), score(s) {}
    };
    
    /**
     * @brief Iterator for basic arcs in a language model.
     *
     * Basic arcs are direct outgoing transitions, i.e. in ngram models they are
     * all possible words given a state excluding back-off transitions.
     *
     * @note This class needs a referenced pointer to a language model.
     */
    class BasicArcIterator {
      AprilUtils::SharedPtr<LMInterface> lm;
      Key key;
      Key arc;
      Score threshold;
      WordScoreTuple word_score;
    public:
      BasicArcIterator(AprilUtils::SharedPtr<LMInterface> lm,
                       Key k, Key a, Score th) :
        lm(lm), key(k), arc(a), threshold(th) {
        word_score = lm->getBasicArc(key, arc);
      }
      
      BasicArcIterator(const BasicArcIterator &other) :
        lm(other.lm), key(other.key), arc(other.arc),
        threshold(other.threshold) {
      }
      
      BasicArcIterator &operator=(const BasicArcIterator &other) {
        lm = other.lm;
        key = other.key;
        arc = other.arc;
        threshold = other.threshold;
      }
      
      BasicArcIterator &operator++() {
        arc = lm->getNextBasicArc(key, arc);
        word_score = lm->getBasicArc(key, arc);
        return *this;
      }
      
      const WordScoreTuple &operator*() const {
        return word_score;
      }
      
      const WordScoreTuple *operator->() const {
        return &word_score;
      }
    };
    
  protected:
    /// Auxiliary result vector.
    AprilUtils::vector<KeyScoreBurdenTuple> result;
    
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
    
    /**
     * @brief Returns an BasicArcIterator for basic arcs outgoing from key state.
     *
     * Basic arcs are direct outgoing transitions, i.e. in ngram models they are
     * all possible words given a state excluding back-off transitions.
     */
    virtual BasicArcIterator beginBasicArcs(Key key, Score threshold) = 0;

    /**
     * @brief Returns an end BasicArcIterator for basic arcs outgoing from key state.
     *
     * Basic arcs are direct outgoing transitions, i.e. in ngram models they are
     * all possible words given a state excluding back-off transitions.
     */
    virtual BasicArcIterator endBasicArcs(Key key) = 0;
    
    /**
     * @brief Returns the next Key refering to a basic arc given context,arc
     * keys
     *
     * This method is used by BasicArcIterator to traverse all arcs from any
     * Key context.
     *
     * @note This method is expected to return an <b>invalid arc</b> when there
     * are no more arcs. This <b>invalid arc</b> should be the same arc used in
     * method endBasicArcs to build the end iterator object.
     */
    virtual Key getNextBasicArc(Key key, Key arc) const = 0;
    
    /**
     * @brief Returns the WordScoreTuple corresponding to the given context,arc
     * keys
     *
     * This method is used by BasicArcIterator to consult for word,score pair
     * corresponding to iterator position.
     *
     * @note An <b>invalid arc</b> would be given to this method, in this case
     * the returned value is undefined and not important. See method
     * getNextBasicArc() for more information.
     */
    virtual WordScoreTuple getBasicArc(Key key, Key arc) const = 0;
    
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
     * @param[in,out] result - An AprilUtils::vector with all the destination
     * states.
     *
     * @param threshold - The threshold for the beam pruning.
     *
     * @note The result vector is not cleared, so all the resulting states will
     * be appended using AprilUtils::vector::push_back() method.
     */
    virtual void get(Key key, WordType word, Burden burden,
                     AprilUtils::vector<KeyScoreBurdenTuple> &result,
                     Score threshold) = 0;
    
    /**
     * @brief Computes only the next keys given a pair (Key,WordType).
     *
     * @note Its default implementation uses get() method and discarding Burden
     * and Score
     *
     * @note The result vector is not cleared, so all the resulting states will
     * be appended using AprilUtils::vector::push_back() method.
     *
     * @param key - The context or state from where the transition starts.
     *
     * @param[in,out] result - An AprilUtils::vector of Key's.
     */
    virtual void getNextKeys(Key key, WordType word,
                             AprilUtils::vector<Key> &result) {
      AprilUtils::vector<KeyScoreBurdenTuple> aux_result;
      get(key, word, Burden(-1,-1), aux_result, Score::zero());
      for (typename AprilUtils::vector<KeyScoreBurdenTuple>::iterator it = aux_result.begin();
           it != aux_result.end(); ++it)
        result.push_back(it->key_score.key);
    }
    
    // -------------- BUNCH MODE -------------
    // Note: we can freely mix insertQuery and insertQueries, but
    // individual queries (gets) and bunch mode queries should not be
    // mixed.

    /**
     * @brief Removes the list of pending queries.
     *
     * @note This must be the first operation to be executed before inserting
     * multiple queries by calling insertQuery() method.
     *
     * @see Methods insertQuery() and getQueries().
     */
    virtual void clearQueries() {
      result.clear();
    }

    /**
     * @brief Call this method for each individual query.
     *
     * @param key - The context or state from where the transition starts.
     *
     * @param word - The transition word.
     *
     * @param burden - The related burden with the current query.
     *
     * @param threshold - A threshold to be use for beam pruning.
     *
     * @note The default implementation uses the get() method within a for loop.
     */
    virtual void insertQuery(Key key, WordType word, Burden burden,
                             Score threshold) {
      get(key,word,burden,result,threshold);
    }
    
    /**
     * @brief A specialization of insertQuery() method for multiple transition
     * words.
     *
     * @param key - The context or state from where the transition starts.
     *
     * @param id_key - A half part of the Burden.
     *
     * @param words - An AprilUtils::vector of WordIdScoreTuple.
     *
     * @param is_sorted - Indicates if the words vector is sorted by word.
     *
     * @note The Burden is computed by the pair (id_key,id_word), where id_key
     * is the given parameter and id_word is different for each word in the
     * given words vector.
     *
     * @note The default implementation uses the insertQuery() method within a
     * for loop.
     */
    virtual void insertQueries(Key key, int32_t id_key,
                               AprilUtils::vector<WordIdScoreTuple> words,
                               bool is_sorted=false) {
      UNUSED_VARIABLE(is_sorted);
      // default behavior
      for (typename AprilUtils::vector<WordIdScoreTuple>::iterator it = words.begin();
        it != words.end(); ++it)
        insertQuery(key, it->word, Burden(id_key, it->id_word), it->score);
    }

    /**
     * @brief This method may perform the computation of language model queries.
     *
     * Some LM implementations can benefit of bunch mode (for instance, NNLMs
     * may profit this feature for grouping all queries associated to the same
     * Key, and can perform several forward steps in bunch or mini-batch
     * mode). Other LMs such as those based on automata (e.g. NgramLiraModel)
     * may perform the LM queries in the insert method so that here they only
     * have to return the result vector.
     *
     * @return An AprilUtils::vector of KeyScoreBurdenTuple with the language
     * model computation result for all the queries inserted with insertQuery()
     * or insertQueries() from the last call to clearQueries().
     */
    virtual const AprilUtils::vector<KeyScoreBurdenTuple> &getQueries() {
      return result;
    }

    /// An upper bound on the best transition probability, usually pre-computed
    /// in the model.
    virtual Score getBestProb() const = 0;

    /// An upper bound on the best transition probability departing from key,
    /// usually pre-computed in the model.
    virtual Score getBestProb(Key k) = 0;

    /// Initial key is the initial state (initial context cue).
    virtual Key getInitialKey() = 0;

    /**
     * @brief Returns the Key corresponding to the zero key concept.
     *
     * The zero key is a special identifier used to represent that the lowest
     * possible level in the automata. In a N-gram language model it is the
     * 0-gram key, so the following transition is a unigram.
     *
     * @param[out] k - The zero key, only valid if the method returns @c true.
     *
     * @return A bool indicating if the model contains or not a zero key
     * identifier.
     *
     * @note This method returns false and does nothing on LMs without zero key,
     * in this case, the @c k parameter is not assigned.
     */
    virtual bool getZeroKey(Key &k) const {
      UNUSED_VARIABLE(k);
      // default behavior
      return false;
    }

    /**
     * @brief Returns the score associated to the probability of being a
     * final state.
     *
     * @param k - The key of the current state.
     *
     * @param threshold - For beam prunning.
     *
     * @return A Score with the probability of @c k being a final state.
     *
     * @note This method is not const since it may depend on get which is not
     * const either.
     */
    virtual Score getFinalScore(Key k, Score threshold) = 0;
  };

  /**
   * @brief The LMModel is the thread-safe part of the LM.
   *
   * This class is where the model data is stored, it is thread-safe and
   * therefore can be shared between multiples threads.
   *
   * @note Score is usually a AprilUtils::log_float.
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
  
  typedef LMInterface<uint32_t,AprilUtils::log_float> LMInterfaceUInt32LogFloat;
  typedef LMModel<uint32_t,AprilUtils::log_float> LMModelUInt32LogFloat;

}; // closes namespace

#endif // LANGUAGE_MODEL_INTERFACE_H
