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

#include "ngram_lira.h"
#include <cstdio>		// print_model for testing
#include "uncommented_line.h"	// read data
#include "error_print.h"        // print errors
#include <cstdlib>		// exit
#include <cstring>		// strcmp

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>           // mmap() is defined in this header
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

namespace language_models {

  // format errors are NOT checked!!!
  NgramLiraModel::NgramLiraModel(FILE *fd,
				 unsigned int expectedVocabularySize,
				 const char *expectedVocabulary[],
				 int fan_out_threshold,
				 bool ignore_extra_words_in_dictionary) :
    ignore_extra_words_in_dictionary(ignore_extra_words_in_dictionary),
    fan_out_threshold(fan_out_threshold),
    is_mmapped(false) {
  
    // these values will be changed later
    transition_words_table    = 0;
    transition_table	      = 0;
    linear_search_table	      = 0;
    backoff_table	      = 0;
    max_out_prob	      = 0;
    first_state_binary_search = 0;

    const int bufferSize = 2048; // fixme: attention to buffer overflow
    char buffer[bufferSize];
    float aux;
    //NgramLiraTransition emptyTransition;
    
    //----------------------------------------------------------------------
    // # number of words and words
    get_uncommented_line(buffer,bufferSize,fd);
    sscanf(buffer,"%u",&vocabularySize);

    // TODO:
    //     if (vocabularySize > emptyTransition.maxWordIndex()) {
    //       ERROR_PRINT2("excessive vocabulary size %u >= %u\n",
    // 		   vocabularySize, emptyTransition.maxWordIndex());
    //       exit(128);
    //     }


    if (!ignore_extra_words_in_dictionary &&
	expectedVocabulary &&
	expectedVocabularySize != vocabularySize) {
      ERROR_PRINT2("Expected vocabularySize %u instead of %u\n",
		   expectedVocabularySize,
		   vocabularySize);
      exit(1);
    }
    for (unsigned int i=0;i<vocabularySize;++i) {
      // we read without allowing comments because the '#' symbol may
      // be part of the lexicon
      get_line(buffer,bufferSize,fd);
      if (expectedVocabularySize) {
	buffer[strlen(buffer)-1] = '\0'; // quitamos el \n
	if (strcmp(buffer,expectedVocabulary[i])!=0) {
	  ERROR_PRINT3("word %u is '%s' instead of '%s'\n",
		       i,buffer,expectedVocabulary[i]);
	  exit(1);
	}
      }
    }

    //----------------------------------------------------------------------
    // # max order of n-gram
    get_uncommented_line(buffer,bufferSize,fd);
    unsigned int n;
    sscanf(buffer,"%u",&n);  
    ngram_value = n;
    if (n >= maxN) {
      fprintf(stderr,"Error %u-grams > %u the max value allowed\n",
	      n,maxN);
      exit(-1);
    }

    //----------------------------------------------------------------------
    // # number of states
    get_uncommented_line(buffer,bufferSize,fd);
    sscanf(buffer,"%u",&num_states);  
  
    //----------------------------------------------------------------------
    // # number of transitions
    get_uncommented_line(buffer,bufferSize,fd);
    sscanf(buffer,"%u",&num_transitions);
    transition_words_table = new WordType[num_transitions];
    transition_table       = new NgramLiraTransition[num_transitions];

    //----------------------------------------------------------------------
    // # bound max trans prob
    get_uncommented_line(buffer,bufferSize,fd);
    sscanf(buffer,"%f",&aux);
    best_prob = log_float(aux);

    //----------------------------------------------------------------------
    // # how many different number of transitions
    get_uncommented_line(buffer,bufferSize,fd);
    sscanf(buffer,"%u",&different_number_of_trans);
    linear_search_table = new LinearSearchInfo[different_number_of_trans+1];

    // # "x y" means x states have y transitions
    // sorted by transitions from lower to upper
    int lss             = 0; // linear_search_size
    int aux_first_state = 0;
    int aux_first_trans = 0;
    int how_many_states;
    int state_fan_out;
    for (unsigned int i=0;i<different_number_of_trans; ++i) {
      get_uncommented_line(buffer,bufferSize,fd);
      sscanf(buffer,"%u%u",&how_many_states,&state_fan_out);
      if (state_fan_out <= fan_out_threshold) { // anyadimos otra entrada en la tabla
	linear_search_table[lss].first_state = aux_first_state;
	linear_search_table[lss].fan_out     = state_fan_out;
	linear_search_table[lss].first_index = aux_first_trans;
	aux_first_state += how_many_states;
	aux_first_trans += state_fan_out*how_many_states;
	lss++;
      }
    }
    linear_search_size        = lss;
    first_state_binary_search = aux_first_state;
    // valores centinela:
    linear_search_table[lss].first_state = aux_first_state;
    linear_search_table[lss].fan_out     = state_fan_out;
    linear_search_table[lss].first_index = aux_first_trans;

    size_first_transition = num_states-first_state_binary_search;
    first_transition      = new unsigned int[size_first_transition + 1];
    first_transition     -= first_state_binary_search;

    //----------------------------------------------------------------------
    // # initial state, final state and lowest state
    get_uncommented_line(buffer,bufferSize,fd);
    sscanf(buffer,"%u%u%u",&initial_state,&final_state,&lowest_state);

    //----------------------------------------------------------------------
    // # state backoff_st 'weight(state->backoff_st)' [max_transition_prob]
    // # backoff_st == -1 means there is no backoff
    backoff_table     = new NgramBackoffInfo[num_states];
    max_out_prob = new log_float[num_states];
    // no hace falta inicializar backoff_table explicitamente, por
    // defecto se pone bo_dest_state a 0 y bo_prob a log_float::zero()
    for (unsigned int i=0;i<num_states; ++i) {
      max_out_prob[i] = log_float::zero();
    }
    for (unsigned int i=0;i<num_states; ++i) {
      int leidos;
      unsigned int orig;
      int   bo_dest;
      float dbackoff,maxOutProb;
      get_uncommented_line(buffer,bufferSize,fd);
      leidos = sscanf(buffer,"%u%d%f%f",
		      &orig,&bo_dest,&dbackoff,&maxOutProb);
      log_float backoff;
      if (bo_dest >= 0) {
	backoff_table[orig].bo_prob       = log_float(dbackoff);
	backoff_table[orig].bo_dest_state = (unsigned int)bo_dest;
      }
      if (leidos == 4) {
	max_out_prob[orig] = log_float(maxOutProb);
      }
    }

    //----------------------------------------------------------------------    
    // # transitions
    // # orig dest word prob
    unsigned int last_state  = 0; // da igual
    for (unsigned int i=0;i<num_transitions; ++i) {
      // # orig dest word prob
      unsigned int orig,dest,word;
      float prob;
      get_uncommented_line(buffer,bufferSize,fd);
      sscanf(buffer,"%u%u%u%f",&orig,&dest,&word,&prob);
      if ((i==0 || orig > last_state) && // new or state is changed
	  orig >= first_state_binary_search) {
	first_transition[orig] = i;
      }
      transition_words_table[i] = word;
      transition_table[i].state = dest;
      transition_table[i].prob  = log_float(prob);
      last_state = orig;
    }
    first_transition[num_states] = num_transitions;
  }

  NgramLiraModel::NgramLiraModel(int vocabularySize, WordType final_word) {
    is_mmapped = false;

    // creates a model with 2 stes, ordered by their fan_out:
    //
    // - state 0 -> final_state has fan_out 0
    //
    // - state 1 -> lowest_state and initial_state at the same time,
    //   the fan_out is vocabularySize, it loops with any word
    //   excepting final_word which is used to go to the final state

    ngram_value     = 1;
    num_states      = 2;
    num_transitions = vocabularySize;

    // the 2 states:
    final_state     = 0;
    initial_state   = 1;
    lowest_state    = 1;

    // best probability to go from each state:
    best_prob       = log_float::one();
    max_out_prob    = new log_float[num_states];
    max_out_prob[0] = log_float::zero();
    max_out_prob[1] = log_float::one();

    // backoff information
    backoff_table                  = new NgramBackoffInfo[num_states];
    backoff_table[0].bo_prob       = log_float::zero();
    backoff_table[0].bo_dest_state = 1;
    backoff_table[1].bo_prob       = log_float::zero();
    backoff_table[1].bo_dest_state = 1;

    // linear search info:
    fan_out_threshold                  = 0;
    linear_search_size                 = 1;
    different_number_of_trans          = 2;
    linear_search_table                = new LinearSearchInfo[different_number_of_trans+1];
    linear_search_table[0].first_state = 0;
    linear_search_table[0].fan_out     = 0;
    linear_search_table[0].first_index = 0;
    // sentinel values:
    linear_search_table[1].first_state = 1;
    linear_search_table[1].fan_out     = 0;
    linear_search_table[1].first_index = 0;

    // information about transitions:
    transition_words_table         = new WordType[num_transitions];
    transition_table               = new NgramLiraTransition[num_transitions];
    first_state_binary_search      = lowest_state;
    size_first_transition          = num_states-first_state_binary_search;
    first_transition               = new unsigned int[size_first_transition + 1];
    first_transition              -= first_state_binary_search;
    first_transition[lowest_state] = 0;
    first_transition[num_states]   = num_transitions;

    // let's generate the transitions:
    for (unsigned int i=0;i<num_transitions; ++i) {
      unsigned int word         = i+1;
      transition_words_table[i] = word;
      transition_table[i].state = (word == final_word) ? final_state : lowest_state;
      transition_table[i].prob  = log_float::one();
    }
  }

  NgramLiraModel::~NgramLiraModel() {
    if (is_mmapped) {
      munmap(filemapped, filesize);
      close(file_descriptor);
    } else {
      delete[] transition_words_table;
      delete[] transition_table;
      delete[] linear_search_table;
      delete[] backoff_table;
      delete[] max_out_prob;
      // restoring pointer:
      first_transition += first_state_binary_search;
      delete[] first_transition;
    }
  }

  void NgramLiraModel::save_binary(const char *filename,
				   unsigned int expectedVocabularySize,
				   const char *expectedVocabulary[]) {
    
    if (expectedVocabularySize != vocabularySize) {
      ERROR_PRINT2("Error expected vocabulary is %d instead of %d\n",
		   vocabularySize,
		   expectedVocabularySize);
      exit(1);
    }

    //--------------------------------------------------
    // trying to open file in write mode
    int f_descr;
    mode_t writemode = S_IRUSR | S_IWUSR | S_IRGRP;
    if ((f_descr = open(filename, O_RDWR | O_CREAT | O_TRUNC, writemode)) < 0) {
      ERROR_PRINT1("Error creating file %s\n",filename);
      exit(1);
    }

    //--------------------------------------------------
    // fill the header
    NgramLiraBinaryHeader header;
    header.magic                     = 12345u;
    header.ngram_value               = ngram_value;
    header.vocabularySize            = vocabularySize;   
    header.initial_state             = initial_state;
    header.final_state               = final_state;
    header.lowest_state              = lowest_state;
    header.num_states                = num_states;
    header.num_transitions           = num_transitions;
    header.different_number_of_trans = different_number_of_trans;
    header.linear_search_size        = linear_search_size;
    header.fan_out_threshold         = fan_out_threshold;
    header.first_state_binary_search = first_state_binary_search;
    header.size_first_transition     = size_first_transition;
    header.best_prob                 = best_prob;

    filesize = sizeof(NgramLiraBinaryHeader);
    header.offset_vocabulary_vector  = filesize;
    header.size_vocabulary_vector    = 0;
    for (unsigned int i=0;i<vocabularySize;++i)
      header.size_vocabulary_vector += strlen(expectedVocabulary[i])+1;
    filesize += header.size_vocabulary_vector;

    header.offset_transition_words_table = filesize;
    header.size_transition_words_table = sizeof(WordType)*num_transitions;
    filesize += header.size_transition_words_table;

    header.offset_transition_table = filesize;
    header.size_transition_table   = sizeof(NgramLiraTransition)*num_transitions;
    filesize += header.size_transition_table;

    header.offset_linear_search_table = filesize;
    header.size_linear_search_table = sizeof(LinearSearchInfo)*(different_number_of_trans+1);
    filesize += header.size_linear_search_table;

    header.offset_first_transition = filesize;
    header.size_first_transition_vector = sizeof(unsigned int)*(size_first_transition + 1);
    filesize += header.size_first_transition_vector;

    header.offset_backoff_table = filesize;
    header.size_backoff_table = sizeof(NgramBackoffInfo)*num_states;
    filesize += header.size_backoff_table;

    header.offset_max_out_prob = filesize;
    header.size_max_out_prob = sizeof(log_float)*num_states;
    filesize += header.size_max_out_prob;

    //----------------------------------------------------------------------
    // make file of desired size:

    // go to the last byte position
    if (lseek(f_descr,filesize-1, SEEK_SET) == -1) {
      ERROR_PRINT1("lseek error, position %u was tried\n",
		   (unsigned int)(filesize-1));
      exit(1);
    }
    // write dummy byte at the last location
    if (write(f_descr,"",1) != 1) {
      ERROR_PRINT("write error\n");
      exit(1);
    }
    // mmap the output file
    if ((filemapped = (char*)mmap(0, filesize,
				  PROT_READ|PROT_WRITE, MAP_SHARED,
				  f_descr, 0))  == (caddr_t)-1) {
      ERROR_PRINT("mmap error\n");
      exit(1);
    }

    // copy different things to disk
    memcpy(filemapped,
	   &header,
	   sizeof(NgramLiraBinaryHeader));

    // copy vocabulary
    char *dest_voc = filemapped+header.offset_vocabulary_vector;
    for (unsigned int i=0;i<vocabularySize;++i) {
      strcpy(dest_voc,expectedVocabulary[i]);
      dest_voc += strlen(expectedVocabulary[i])+1;
    }

    memcpy(filemapped + header.offset_transition_words_table,
	   transition_words_table,
	   header.size_transition_words_table);

    memcpy(filemapped + header.offset_transition_table,
	   transition_table,
	   header.size_transition_table);

    memcpy(filemapped+header.offset_linear_search_table,
	   linear_search_table,
	   header.size_linear_search_table);

    memcpy(filemapped+header.offset_first_transition,
	   first_transition + first_state_binary_search,
	   header.size_first_transition_vector);

    memcpy(filemapped+header.offset_backoff_table,
	   backoff_table,
	   header.size_backoff_table);

    memcpy(filemapped+header.offset_max_out_prob,
	   max_out_prob,
	   header.size_max_out_prob);

    // work done, free the resources ;)
    if (munmap(filemapped, filesize) == -1) {
      ERROR_PRINT("munmap error\n");
      exit(1);
    }
    close(f_descr);
  }

  // constructor for binary mmaped data
  NgramLiraModel::NgramLiraModel(const char *filename,
				 unsigned int expectedVocabularySize,
				 const char *expectedVocabulary[],
				 bool ignore_extra_words_in_dictionary) :
    ignore_extra_words_in_dictionary(ignore_extra_words_in_dictionary) {
    //----------------------------------------------------------------------
    // open file:
    if ((file_descriptor = open(filename, O_RDONLY)) < 0) {
      ERROR_PRINT1("Error opening file \"%s\"\n",filename);
      exit(1);
    }
    // find size of input file
    struct stat statbuf;
    if (fstat(file_descriptor,&statbuf) < 0) {
      ERROR_PRINT("Error guessing filesize\n");
      exit(1);
    }
    // mmap the input file
    filesize = statbuf.st_size;
    if ((filemapped = (char*)mmap(0, filesize,
				  PROT_READ, MAP_SHARED,
				  file_descriptor, 0))  ==(caddr_t)-1) {
      ERROR_PRINT("Error mmaping\n");
      exit(1);
    }
    is_mmapped = true;

    NgramLiraBinaryHeader *header = (NgramLiraBinaryHeader *) filemapped;
    if (header->magic != 12345u) {
      ERROR_PRINT("Error magic value, endianism problem?\n");
      exit(1);
    }
    ngram_value               = header->ngram_value;
    vocabularySize            = header->vocabularySize;   
    initial_state             = header->initial_state;
    final_state               = header->final_state;
    lowest_state              = header->lowest_state;
    num_states                = header->num_states;
    num_transitions           = header->num_transitions;
    different_number_of_trans = header->different_number_of_trans;
    linear_search_size        = header->linear_search_size;
    fan_out_threshold         = header->fan_out_threshold;
    first_state_binary_search = header->first_state_binary_search;
    size_first_transition     = header->size_first_transition;
    best_prob                 = header->best_prob;

    // checking vocabulary:
    if (expectedVocabulary) {
      if (!ignore_extra_words_in_dictionary) {
	if (expectedVocabularySize != vocabularySize) {
	  ERROR_PRINT2("Expected vocabularySize %u instead of %u\n",
		       expectedVocabularySize,
		       vocabularySize);
	  exit(1);
	}
      }
      char *dest_voc = filemapped+header->offset_vocabulary_vector;
      for (unsigned int i=0;i<vocabularySize;++i) {
	if (strcmp(dest_voc,expectedVocabulary[i])!=0) {
	  ERROR_PRINT3("word %u is '%s' instead of '%s'\n",
		       i,dest_voc,expectedVocabulary[i]);
	  exit(1);
	}
	dest_voc += strlen(expectedVocabulary[i])+1;
      }
    }

    // assign vector pointers:
    transition_words_table = (WordType *)(filemapped + header->offset_transition_words_table);
    transition_table       = (NgramLiraTransition *)(filemapped + header->offset_transition_table);
    linear_search_table    = (LinearSearchInfo*)(filemapped + header->offset_linear_search_table);
    backoff_table          = (NgramBackoffInfo*)(filemapped + header->offset_backoff_table);
    max_out_prob           = (log_float*)(filemapped + header->offset_max_out_prob);
    first_transition       = (unsigned int*)(filemapped + header->offset_first_transition) - first_state_binary_search;

    // at this point, all seems to be ok :)
  }



  log_float NgramLiraInterface::get(const uint32_t &state, int32_t idState,
				    int32_t word, int32_t idWord,
				    vector<KeyScoreIdTuple> &result,
				    log_float threshold) {

    log_float accum_backoff = log_float::one();
    unsigned int st         = state;
    int       depth         = 0; // number of times we have performed backoff
    int       linear_index;

    for (;;) {
      if (depth > preparedLevel) {
	if (st >= first_state_binary_search) {
	  linear_index = -1; // indicates dichotomic search
	} else { // let's look the LinearSearchInfo of this state
	  linear_index = 0;
	  while(linear_search_table[linear_index].first_state <= st)
	    linear_index++;
	  linear_index--; // nos habiamos pasado ;)
	}
	linear_index_vector[depth] = linear_index;
	preparedLevel              = depth;
      } else {
	linear_index = linear_index_vector[depth];
      }
      
      if (linear_index >= 0) {
	LinearSearchInfo *info = &linear_search_table[linear_index];
	// range of transitions during the search:
	unsigned int first_tr_index = (st - info->first_state)*info->fan_out + info->first_index;
	unsigned int last_tr_index  = first_tr_index + info->fan_out;
	// lineal search:
	for (unsigned int tr_index = first_tr_index; tr_index < last_tr_index; tr_index++)
	  if (transition_words_table[tr_index] == word) {
	    result.clear();
	    result.push_back(KeyScoreIdTuple(transition_table[tr_index].state,
					     accum_backoff * transition_table[tr_index].prob,
					     idState,
					     idWord));
	    return 1; // result.size();
	  }
      } else {
	// the dichotomic search of the transition index is not based
	// on the binary_search template in order to be able to return
	// as soon as the word is found
	unsigned int left  = first_transition[st];
	unsigned int right = first_transition[st+1] - 1;
	while (left <= right) {
	  unsigned int tr_index     = (left+right)/2;
	  unsigned int current_word = transition_words_table[tr_index];
	  if (current_word == word) {
	    fkey.set(transition_table[tr_index].state);
	    return accum_backoff * transition_table[tr_index].prob;
	  } else if (current_word < word) {
	    left  = tr_index+1;
	  } else {
	    right = tr_index-1;
	  }
	}
      }
      // apply backoff when the transition is not found:
      accum_backoff *= backoff_table[st].bo_prob;
      st             = backoff_table[st].bo_dest_state;      
      depth++;
    }
    assert(false && "this should never happen");
    return log_float::zero(); // evitar warning compilador
  }

  void NgramLiraInterface::clearQueries() {
    preparedLevel = -1;
    result.clear();
  }

  // void NgramLiraInterface::insertQueries(const Key &key, int32_t idKey,
  // 					 vector<WordIdScoreTuple> words) {
  //   // todo: we can take profit of the lira data structure to improve
  //   // the efficiency of this method w.r.t. the naive implementation
  // }

  // internal method used by findKeyFromNgram
  unsigned int NgramLiraInterface::getDestState(unsigned int st, unsigned int word) {
    do {
      if (st < first_state_binary_search) {
	// linear search, we look for the LinearSearchInfo of st
	int linear_index = 0;
	while(linear_search_table[linear_index].first_state <= st)
	  linear_index++;
	// -1 because the search stopped too late ;)
	LinearSearchInfo *info = &linear_search_table[linear_index-1];
	// range of transitions during the search:
	unsigned int first_tr_index = ((st - info->first_state)*info->fan_out +
				       info->first_index);
	unsigned int last_tr_index  = first_tr_index + info->fan_out;
	// lineal search:
	for (unsigned int tr_index = first_tr_index; tr_index < last_tr_index; tr_index++) {
	  if (transition_words_table[tr_index] == word) {
	    return transition_table[tr_index].state;
	  }
	}
      } else {
	// the dichotomic search of the transition index is not based
	// on the binary_search template in order to be able to return
	// as soon as the word is found
	unsigned int left  = first_transition[st];
	unsigned int right = first_transition[st+1] - 1;
	while (left <= right) {
	  unsigned int tr_index     = (left+right)/2;
	  unsigned int current_word = transition_words_table[tr_index];
	  if (current_word == word) {
	    return transition_table[tr_index].state;
	  } else if (current_word < word) {
	    left  = tr_index+1;
	  } else {
	    right = tr_index-1;
	  }
	}
      }
      // apply backoff when the transition is not found:
      st = backoff_table[st].bo_dest_state;      
    } while (1);
    return st;
  }

  unsigned int NgramLiraInterface::findKeyFromNgram(const unsigned int *wordSequence,
						    int len) {
    int st = lowest_state;
    for (int i=0; i<len; ++i)
      if (wordSequence[i] > 0)
	st = getDestState(st,wordSequence[i]);
    return st;
  }

} // closes namespace language_models
