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
#ifndef FEATURE_BASED_LM_H
#define FEATURE_BASED_LM_H

#include <stdint.h>
#include "april_assert.h"
#include "bunch_hashed_LM.h"
#include "error_print.h"
#include "function_interface.h"
#include "history_based_LM.h"
#include "identity_function.h"
#include "logbase.h"
#include "smart_ptr.h"
#include "token_vector.h"
#include "trie_vector.h"
#include "unused_variable.h"
#include "vector.h"

namespace LanguageModels {

  /**
   * @brief A LM base class for LMs which depend on a history of words (like
   * Ngrams) and can work in bunch mode.
   *
   * A FeatureBasedLM depends into two function filter instances, both
   * derived from Functions::FunctionInterface class. The corresponding interface
   * class is FeatureBasedLMInterface and it is ready to work with a bunch
   * of queries. Every bunch is traversed and filtered in order to extract
   * new features from the initial word identifiers. Two different kind of
   * filters has been defined:
   *   1. QueryFilters receive a query formed by a context words sequence
   *      (\f$h\f$) and a list of next words (\đ$w_1 w_2 \ldots\f$). This query
   *       summarizes the computation of \f$p(w1|h), p(w2|h), \ldots\f$.
   *   2. BunchFiltes receive a bunch of queries like such above.
   * The filters can modify the given information in many different ways,
   * allowing to produce Ngram Skips, ShortLists, FactoreBased approaches, among
   * others.
   * 
   * @see FeatureBasedLM::query_filter and FeatureBasedLM::bunch_filter
   * properties for more documentation.
   */  
  template <typename Key, typename Score>
  class FeatureBasedLM;

  /**
   * @brief Interface part of FeatureBasedLM class.
   * @see FeatureBasedLM for more documentation.
   */
  template <typename Key, typename Score>
  class FeatureBasedLMInterface : public HistoryBasedLMInterface <Key,Score>,
                                  public BunchHashedLMInterface <Key, Score> {
    friend class FeatureBasedLM<Key,Score>;

    AprilUtils::SharedPtr<Functions::FunctionInterface> query_filter;
    AprilUtils::SharedPtr<Functions::FunctionInterface> bunch_filter;

  protected:
    FeatureBasedLMInterface(FeatureBasedLM<Key,Score>* model) :
      HistoryBasedLMInterface<Key,Score>(model),
      BunchHashedLMInterface<Key,Score>(model),
      query_filter(model->getQueryFilter()),
      bunch_filter(model->getBunchFilter()) { }

    typedef typename BunchHashedLMInterface<Key,Score>::KeyWordHash KeyWordHash;
    typedef typename BunchHashedLMInterface<Key,Score>::WordResultHash WordResultHash;
    typedef typename BunchHashedLMInterface<Key,Score>::KeyScoreMultipleBurdenTuple KeyScoreMultipleBurdenTuple;

    /**
     * @brief Computes the scores for all the given queries.
     *
     * The @c queries_bunch_token is an instance of Basics::TokenBunchVector
     * which contains several instances of Basics::TokenBunchVector, as many
     * as size of the bunch. Every instance has two components:
     *   1. A Basics::TokenVectorUint32 for the context words.
     *   2. A Basics::TokenVectorUint32 for the next words. All the words in
     *      this vector share the previous context words.
     *
     * @param queries_bunch_token - An input Token with all the queries.
     * @param[out] scores - A vector where computes scores will be stored.
     *
     * @note The @c scores vector is only updated by calling to
     * AprilUtils::vector::push_back() method, allowing to chain multiple calls
     * into the same @c scores vector.
     */
    virtual void executeQueries(Basics::Token *queries_bunch_token,
                                AprilUtils::vector<Score> &scores) = 0;
    
    /**
     * @brief Generates the input expected by executeQueries() method.
     */
    virtual void computeKeysAndScores(KeyWordHash &ctxt_hash,
                                      unsigned int bunch_size) {
      april_assert(sizeof(WordType) == sizeof(uint32_t));
      const int order = getLMModel()->ngramOrder();
      april_assert(order != -1);
      AprilUtils::SharedPtr<Basics::TokenBunchVector> 
        queries_bunch_token( new Basics::TokenBunchVector() );
      AprilUtils::vector<Score> scores;

      // For each context key entry
      for (typename KeyWordHash::iterator it = ctxt_hash.begin();
           it != ctxt_hash.end(); ++it) {
        Key context_key = it->first;
        WordResultHash &word_hash = it->second;
        // offset init'd to 0
        unsigned int offset = 0;
        
        // The procedure needs to create a token where context words will be
        // stored. This token has a vector container which needs to resized
        // before call to getContextProperties().
        AprilUtils::SharedPtr<Basics::TokenVectorUint32>
          context_words_token( new Basics::TokenVectorUint32() );
        AprilUtils::vector<WordType> &context_words_vector =
          context_words_token->getContainer();
        context_words_vector.resize( order - 1 );
        // the next pointer will be given to getContextProperties() as output
        WordType *context_words = context_words_vector.begin();
        const unsigned int context_size = this->getContextProperties(context_key,
                                                                     context_words,
                                                                     offset);

        AprilUtils::SharedPtr<Basics::TokenVectorUint32>
          next_words_token( new Basics::TokenVectorUint32() );
        // For each word entry
        for (typename WordResultHash::iterator it2 = word_hash.begin();
             it2 != word_hash.end(); ++it2) {
          WordType word = it2->first;
          KeyScoreMultipleBurdenTuple &result_tuple = it2->second;
          
          // First pass we get the next key
          // collect context and word tokens
          result_tuple.key_score.key =
            this->getDestinationKey(context_words,
                                    offset,
                                    context_size,
                                    word);
          
          next_words_token->push_back(word);
        }

        // Put together context and next word tokens
        AprilUtils::SharedPtr<Basics::TokenBunchVector>
          query_token( new Basics::TokenBunchVector() );
        query_token->getContainer().reserve(2); // 2 items
        query_token->push_back(context_words_token.get()); // [0]
        query_token->push_back(next_words_token.get());    // [1]
        
        // Apply query filter
        AprilUtils::SharedPtr<Basics::Token> filtered_query_token;
        filtered_query_token = query_filter->calculate(query_token.get());
        
        // Put the current query into the bunch of queries
        queries_bunch_token->push_back(filtered_query_token.get());
        
        // If we have a full bunch, process it
        if (query_token->size() % bunch_size == 0) {
          // Apply bunch filter
          AprilUtils::SharedPtr<Basics::Token>
            filtered_queries_bunch_token( bunch_filter->calculate(queries_bunch_token.get()) );
          executeQueries(filtered_queries_bunch_token.get(), scores);
          queries_bunch_token->clear();
        }
      }
      
      // If there is something left in the bunch, process it
      if (queries_bunch_token->size() > 0) {
        // Apply bunch filter
        AprilUtils::SharedPtr<Basics::Token>
          filtered_queries_bunch_token( bunch_filter->calculate(queries_bunch_token.get()) );
        executeQueries(filtered_queries_bunch_token.get(), scores);
        queries_bunch_token->clear();
      }
      
      // FIXME: find a way to store scores into hash table during the first pass

      // Second pass to store scores at table
      unsigned int k = 0;
      for (typename KeyWordHash::iterator it = ctxt_hash.begin();
        it != ctxt_hash.end(); ++it) {
        Key context_key = it->first;
        WordResultHash &word_hash = it->second;

        for (typename WordResultHash::iterator it2 = word_hash.begin();
          it2 != word_hash.end(); ++it2) {
          KeyScoreMultipleBurdenTuple &result_tuple = it2->second;
          result_tuple.key_score.score = scores[k++];
        }
      }
    }

  public:
    virtual ~FeatureBasedLMInterface() { }
    
    void incRef() {
      HistoryBasedLMInterface<Key,Score>::incRef();
      BunchHashedLMInterface<Key,Score>::incRef();
    }

    bool decRef() {
      HistoryBasedLMInterface<Key,Score>::decRef();
      BunchHashedLMInterface<Key,Score>::decRef();
      return (HistoryBasedLMInterface<Key,Score>::getRef() <= 0);
    }

    Basics::Token* applyQueryFilter(Basics::Token* token) {
      return query_filter->calculate(token);
    }

    Basics::Token* applyBunchFilter(Basics::Token* token) {
      return query_filter->calculate(token);
    }

    virtual LMModel<Key, Score>* getLMModel() {
      return HistoryBasedLMInterface<Key,Score>::model;
    }
  };
  
  // Documentation is in the forward declaration in the top of this file.
  template <typename Key, typename Score>
  class FeatureBasedLM : public HistoryBasedLM <Key,Score>,
                         public BunchHashedLM <Key,Score> {
  private:
    
    /**
     * @brief A function which will be used to filter every LM query.
     *
     * The @c query_filter expects an instance of Basics::TokenBunchVector
     * which contains two components:
     *   1. A Basics::TokenVectorUint32 for the context words.
     *   2. A Basics::TokenVectorUint32 for the next words. All the words in
     *      this vector share the previous context words.
     * The query_filter can modify any of both components **in-place** or return
     * a new allocated Basics::TokenBunchVector instance.
     */
    AprilUtils::SharedPtr<Functions::FunctionInterface> query_filter;
    
    /**
     * @brief A function which will be used to filter a bunch of LM queries.
     *
     * @see LanguageModels::FeatureBasedLMInterface::executeQueries() method
     * for documentation about the expected input token format.
     */
    AprilUtils::SharedPtr<Functions::FunctionInterface> bunch_filter;

  public:
    FeatureBasedLM(int ngram_order,
                   WordType init_word,
                   AprilUtils::TrieVector *trie_vector,
                   unsigned int bunch_size,
                   Functions::FunctionInterface *query_filter,
                   Functions::FunctionInterface *bunch_filter=0) :
      HistoryBasedLM<Key,Score>(ngram_order,
                                init_word,
                                trie_vector),
      BunchHashedLM<Key,Score>(bunch_size),
      query_filter(query_filter),
      bunch_filter(bunch_filter) {
      if (this->bunch_filter.empty()) {
        bunch_filter = new Functions::IdentityFunction();
      }
      april_assert(!this->query_filter.empty());
      april_assert(!this->bunch_filter.empty());
    }

    virtual ~FeatureBasedLM() {
    }

    Functions::FunctionInterface* getQueryFilter() {
      return query_filter.get();
    }

    Functions::FunctionInterface* getBunchFilter() {
      return bunch_filter.get();
    }

    void incRef() {
      HistoryBasedLM<Key,Score>::incRef();
      BunchHashedLM<Key,Score>::incRef();
    }

    bool decRef() {
      HistoryBasedLM<Key,Score>::decRef();
      BunchHashedLM<Key,Score>::decRef();
      return (HistoryBasedLM<Key,Score>::getRef() <= 0);
    }
  };

  typedef FeatureBasedLMInterface<uint32_t, AprilUtils::log_float>
  FeatureBasedLMInterfaceUInt32LogFloat;
  typedef FeatureBasedLM<uint32_t, AprilUtils::log_float>
  FeatureBasedLMUInt32LogFloat;
}; // closes namespace

#endif // FEATURE_BASED_LM_H
