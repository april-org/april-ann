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
    
    /**
     * @brief Arcs iterator in a language model.
     *
     * Language models can different kinds of iterators for traverse different
     * arc kinds. For instance, in ngram models, once can traverse all arcs
     * outgoing from a given state ignoring backoff transitions. On the other
     * hand, once can traverse only backoff transitions, ignoring non-backoff
     * transitions. Language models implement different methods in final
     * classes, allowing to traverse the desired kind of arcs. The iterators are
     * generic, allowing to export them in the same way into Lua. An abstract
     * class, ArcsIterator::StateControl, is used to declare the interface
     * between the iterator and the language model. Derived classes from
     * LMInterface are required to implement properly the StateControl class.
     */
    class ArcsIterator {
    public:
      
      /**
       * @brief This class implements the basic API needed for traverse
       * languages models using ArcsIterator class.
       *
       * @note The API receives a LMInterface pointer in methods which needs
       * to access LM data structures. So, minimum state is required in this
       * class, just to represent a position in the language model. The caller
       * is responsible to perform the call with the proper LM pointer.
       */
      class StateControl {
        friend class ArcsIterator;
      public:

        virtual ~StateControl() {}
      private:
        /**
         * @brief This method traverses until next arc which is above threshold
         *
         * @note This method will update the state with <b>end</b> iterator state
         * when necessary.
         */
        virtual void moveToNext(LMInterface *lm, Score threshold) = 0;
        
        /**
         * @brief This method returns a word
         *
         * @note If this method is called when state is <b>end</b>, the returned
         * value is undefined.
         */
        virtual WordType getWord(LMInterface *lm) = 0;
        
        /// Compares two StateControl objects.
        virtual bool equals(const StateControl *other) const = 0;
        
        /// Copies other into this.
        virtual void copy(const StateControl *other) = 0;
        
        /// Returns a deep copy of this.
        virtual StateControl *clone() const = 0;
        
        /// Indicates if we achieved the end.
        virtual bool isEnd(LMInterface *lm) const = 0;
      };
      
    private:
      LMInterface *lm;
      Score threshold;
      AprilUtils::UniquePtr<StateControl> state;
      
    public:
      ArcsIterator() {}
      ArcsIterator(ArcsIterator const &other) :
        lm(other.lm), threshold(other.threshold), state(other.state->clone()) {}

      ArcsIterator(LMInterface *lm, Score th,
                   AprilUtils::UniquePtr<StateControl> state) :
        lm(lm), threshold(th), state(state) {}
      
      virtual bool operator!=(const ArcsIterator &other) const {
        return (*this) != other;
      }
      virtual bool operator==(const ArcsIterator &other) const {
        return ( this->lm==other.lm &&
                 this->threshold==other.threshold &&
                 state->equals(other.state.get()) );
      }
      virtual ArcsIterator &operator=(ArcsIterator const &other) {
        lm = other.lm;
        threshold = other.threshold;
        state->copy(other.state.get());
        return *this;
      }
      virtual ArcsIterator &operator++() {
        state->moveToNext(lm, threshold);
        return *this;
      }
      virtual WordType operator*() {
        return state->getWord(lm);
      }
      virtual bool isEnd() const {
        return state->isEnd(lm);
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
    
    // /**
    //  * @brief Returns an ArcsIterator for basic arcs outgoing from key state.
    //  *
    //  * This method returns an iterator to the first transition with probability
    //  * above the given threshold.
    //  *
    //  * @see ArcsIterator class
    //  */
    // virtual ArcsIterator beginArcs(Key key, Score threshold) = 0;

    // /**
    //  * @brief Returns an end ArcsIterator for basic arcs outgoing from key state.
    //  *
    //  * This method returns the first invalid iterator related with the given
    //  * key.
    //  *
    //  * @see ArcsIterator class
    //  */
    // virtual ArcsIterator endArcs(Key key) = 0;
    
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
