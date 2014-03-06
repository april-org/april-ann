/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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

#ifndef NGRAM_LIRA_H
#define NGRAM_LIRA_H

#include "LM_interface.h"
#include "binary_search.h"
#include <cmath>
#include <cassert>
#include <climits> // UINT_MAX
#include "logbase.h"

using namespace april_utils;
using april_utils::binary_search;

namespace language_models {

  struct NgramLiraTransition {
    unsigned int state;
    log_float	 prob;
  };

  /// lambda backoff transition:
  struct NgramBackoffInfo {
    unsigned int bo_dest_state; // estado al que se baja
    log_float    bo_prob; // probabilidad de bajar al estado bo_dest_state
    NgramBackoffInfo() : bo_dest_state(0), bo_prob(log_float::zero()) { }
  };

  /// states are sorted by fan_out in .lira format
  struct LinearSearchInfo {
    unsigned int first_state;// first state with a given fan_out
    unsigned int fan_out;    // number of output transitions of states
    unsigned int first_index;// index of first transition of first_state
  };

  struct NgramLiraBinaryHeader {
    unsigned int magic;
    unsigned int ngram_value;
    unsigned int vocabularySize;
    unsigned int initial_state;
    unsigned int final_state;
    unsigned int lowest_state;
    unsigned int num_states;
    unsigned int num_transitions;
    unsigned int different_number_of_trans; // size del linear_search_table
    unsigned int linear_search_size;
    unsigned int fan_out_threshold;
    unsigned int first_state_binary_search;
    unsigned int size_first_transition;
    log_float    best_prob;
    size_t       size_first_transition_vector;
    size_t       offset_vocabulary_vector;
    size_t       size_vocabulary_vector;
    size_t       offset_transition_words_table;
    size_t       size_transition_words_table;
    size_t       offset_transition_table;
    size_t       size_transition_table;
    size_t       offset_linear_search_table;
    size_t       size_linear_search_table;
    size_t       offset_first_transition;
    size_t       offset_backoff_table;
    size_t       size_backoff_table;
    size_t       offset_max_out_prob;
    size_t       size_max_out_prob;
  };

  class NgramLiraModel : public LMModel<uint32_t,log_float> {
  public:

    // allows the dictionary used to check the model to be larger than
    // the actual list of words in the model
    bool ignore_extra_words_in_dictionary;

    unsigned int vocabularySize;
    unsigned int initial_state;
    unsigned int final_state;
    unsigned int lowest_state;
    unsigned int num_states;
    unsigned int num_transitions;
    unsigned int ngram_value;


    // transitions table is divided in two parts, both of them are of
    // size num_transitions:
    WordType *transition_words_table;      ///< size num_transitions
    NgramLiraTransition *transition_table; ///< size num_transitions
    NgramBackoffInfo *backoff_table; ///< size num_states

    // upper bound on the max probability outgoing from each state
    log_float *max_out_prob; ///< size num_states, when fan_out==0 equals zero

    // number of different fan outs
    unsigned int different_number_of_trans; // size del linear_search_table
    LinearSearchInfo *linear_search_table; ///< size different_number_of_trans

    /// threshold where to stop searching in array linear_search_table
    unsigned int linear_search_size;

    /// fan_out_threshold the best out degree of a state whose
    /// transitions are linearly searched
    unsigned int fan_out_threshold;
    /// index of first state whose transitions are no longer linearly searched
    unsigned int first_state_binary_search;

    unsigned int  size_first_transition; ///< num_states-first_state_binary_search
    unsigned int *first_transition;     ///< vector shifted first_state_binary_search

    /// best_prob is the max of max_tr_prob_table
    log_float best_prob; ///< loaded from .lira

    //----------------------------------------------------------------------
    // data in case vectors are mapped:
    bool   is_mmapped;
    int    file_descriptor;
    size_t filesize;
    char  *filemapped;
    //----------------------------------------------------------------------
    
    ~NgramLiraModel();

    // generates the binary data useful for mmaped version
    void save_binary(const char *filename,
		     unsigned int expectedVocabularySize,
		     const char *expectedVocabulary[]);
    
    // constructor for binary mmaped data
    NgramLiraModel(const char *filename,
		   unsigned int expectedVocabularySize,
		   const char *expectedVocabulary[],
		   bool ignore_extra_words_in_dictionary);

    /// fan_out_threshold is used to distinguish automata states when
    /// looking for their transitions
    /// expectedVocabularySize is an array of char* strings, the
    /// vocabulary is not checked when this argument is NULL
    NgramLiraModel(FILE *fd,
		   unsigned int expectedVocabularySize,
		   const char *expectedVocabulary[],
		   int fan_out_threshold,
		   bool ignore_extra_words_in_dictionary);

    // creates a simple model with two states: one which is a loop and
    // a final state
    NgramLiraModel(int vocabularySize, WordType final_word);


    bool isDeterministic() {
      return true;
    }

    int ngramOrder() { return ngram_value; }
    
    bool requireHistoryManager() { return false; }

    LMInterface<Key,Score>* getInterface(LMHistoryManager *hmanager=0) {
      return new NgramLiraInterface(this);
    }

    
  }; // closes class NgramLiraModel

  class NgramLiraInterface : public LMInterface<uint32_t,log_float> {
    NgramLiraModel *liraModel;

    // used to prepare, this class is not thread safe :'(
    static const unsigned int maxN = 20;
    int preparedLevel;

    // values contained in linear_index_vector are useful until prepareLevel
    int linear_index_vector[maxN];

  public:

    NgramLiraInterface(NgramLiraModel *liraModel) :
      liraModel(liraModel) {
      IncRef(liraModel);
    }
    
    ~NgramLiraInterface() {
      DecRef(liraModel);
    }
    
    LMModel* getLMModel() {
      // it is the responsibility of the receiver to IncRef the model
      return liraModel;
    }
    
    int get(const Key &key, int32_t idKey,
	    WordType word, int32_t idWord,
	    vector<KeyScoreIdTuple> &result, Score threshold);
    
    void clearQueries();
    
    void insertQuery(const Key &key, int32_t idKey,
		     Word word, int32_t idWord,
		     Score threshold);
    
    void insertQueries(const Key &key, int32_t idKey,
		       vector<WordIdScoreTuple> words);
    
    log_float    getBestProb()                   { return best_prob; }
    log_float    getBestProb(unsigned int &k)    { return max_out_prob[k]; }
    void getZeroGramKey(unsigned int initial_word, unsigned int &k) {
      k = lowest_state;
    }
    void getInitialKey(unsigned int initial_word, unsigned int &k) {
      k = initial_state;
    }
    void getFinalKey(unsigned int final_word, unsigned int &k) {
      k = final_state;
    }

    // useful methods to look for an state:
    unsigned int getDestState(unsigned int st, unsigned int word);
    unsigned int findKeyFromNgram(const unsigned int *wordSequence, int len);
    
  };

} // closes namespace language_models

#endif // NGRAM_LIRA_H
