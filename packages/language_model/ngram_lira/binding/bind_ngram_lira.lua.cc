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

//BIND_HEADER_C
#include "bind_april_io.h"
#include "bind_LM_interface.h"
#include "bind_util.h"
using namespace AprilUtils;
using namespace AprilIO;
//BIND_END

//BIND_HEADER_H
#include "bind_LM_interface.h"
#include "history_based_LM.h"
#include "history_based_ngram_lira.h"
#include "bunch_hashed_LM.h"
#include "bunch_hashed_ngram_lira.h"
#include "ngram_lira.h"
using namespace LanguageModels;
//BIND_END

//BIND_LUACLASSNAME LMModelUInt32LogFloat language_models.model
//BIND_LUACLASSNAME LMInterfaceUInt32LogFloat language_models.interface

//BIND_LUACLASSNAME HistoryBasedLMUInt32LogFloat language_models.history_based_model
//BIND_LUACLASSNAME HistoryBasedLMInterfaceUInt32LogFloat language_models.history_based_interface
//BIND_LUACLASSNAME BunchHashedLMUInt32LogFloat language_models.bunch_hashed_model
//BIND_LUACLASSNAME BunchHashedLMInterfaceUInt32LogFloat language_models.bunch_hashed_interface

///////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME NgramLiraModel ngram.lira.model
//BIND_CPP_CLASS    NgramLiraModel
//BIND_SUBCLASS_OF  NgramLiraModel LMModelUInt32LogFloat

//BIND_LUACLASSNAME NgramLiraInterface ngram.lira.interface
//BIND_CPP_CLASS    NgramLiraInterface
//BIND_SUBCLASS_OF  NgramLiraInterface LMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR NgramLiraModel
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, 
		     "binary", // boolean
                     "stream", // opens a stream
		     "filename",
		     "vocabulary",
		     "final_word", 
		     "fan_out_threshold",
		     // sirve para permitir que el diccionario que te
		     // pasan sea mas grande que el que hay en el
		     // modelo lira
		     "ignore_extra_words_in_dictionary",
		     (const char *)0); // 0 para terminar el check_table_fields
  //
  bool ignore_extra_words_in_dictionary = false;
  bool binary = false;
  const char *filename;
  SharedPtr<StreamInterface> stream;
  NgramLiraModel *obj=0;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, binary,
				       bool,
				       binary, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, filename, string, filename, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, stream, AuxStreamInterface<StreamInterface>, stream, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, ignore_extra_words_in_dictionary,
				       bool,
				       ignore_extra_words_in_dictionary, false);
  lua_getfield(L,1,"vocabulary");
  if (lua_isnil(L,-1)) {
    LUABIND_ERROR("error ngram.lira.model constructor requires vocabulary table");
  }
  int vocabulary_size, final_word;
  LUABIND_GET_TABLE_PARAMETER(1, final_word, int, final_word);
  LUABIND_TABLE_GETN(-1, vocabulary_size);
  const char **vocabulary_vector = new const char *[vocabulary_size];
  LUABIND_TABLE_TO_VECTOR(-1, string, vocabulary_vector, vocabulary_size);

  if (binary) {

    obj=new NgramLiraModel(filename,
			   (unsigned int)vocabulary_size,
			   vocabulary_vector,
			   final_word,
			   ignore_extra_words_in_dictionary);
  } else {
    int fan_out_threshold;
    LUABIND_GET_TABLE_PARAMETER(1, fan_out_threshold, int, fan_out_threshold);
    if (stream.empty()) {
      stream = new FileStream(filename, "r");
      if (stream->hasError()) {
        LUABIND_FERROR1("unable to open %s\n",filename);
      }
    }
    obj=new NgramLiraModel(stream.get(),
			   (unsigned int)vocabulary_size,
			   vocabulary_vector,
			   final_word,
			   fan_out_threshold,
			   ignore_extra_words_in_dictionary);
  }
  delete[] vocabulary_vector;
  LUABIND_RETURN(NgramLiraModel, obj);    
}
//BIND_END

//BIND_DESTRUCTOR NgramLiraModel
{
}
//BIND_END

//BIND_METHOD NgramLiraModel save_binary
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, 
		     "filename",
		     "vocabulary",
		     (const char *)0); // 0 para terminar el check_table_fields
  //
  const char *filename;
  LUABIND_GET_TABLE_PARAMETER(1, filename, string, filename);
  lua_getfield(L,1,"vocabulary");
  if (lua_isnil(L,-1)) {
    LUABIND_ERROR("error save_binary method requires vocabulary table");
  }
  int vocabulary_size;
  LUABIND_TABLE_GETN(-1, vocabulary_size);
  const char **vocabulary_vector = new const char *[vocabulary_size];
  LUABIND_TABLE_TO_VECTOR(-1, string, vocabulary_vector, vocabulary_size);
  
  obj->saveBinary(filename,
		  (unsigned int)vocabulary_size,
		  vocabulary_vector);
  delete[] vocabulary_vector;
}
//BIND_END

//BIND_CLASS_METHOD NgramLiraModel loop
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, 
		     "vocabulary_size",
		     "final_word", 
		     (const char *)0); // 0 para terminar el check_table_fields
  //
  int vocabulary_size, final_word;
  LUABIND_GET_TABLE_PARAMETER(1,
			      vocabulary_size, 
			      int,
			      vocabulary_size);
  LUABIND_GET_TABLE_PARAMETER(1,
			      final_word, 
			      int,
			      final_word);
  NgramLiraModel *obj=new NgramLiraModel((unsigned int)vocabulary_size,
					 (unsigned int)final_word);
  LUABIND_RETURN(NgramLiraModel, obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR NgramLiraInterface
{
  LUABIND_ERROR("Use the model method get_interface");
}
//BIND_END

//BIND_METHOD NgramLiraInterface non_backoff_arcs_iterator
{
  unsigned int key;
  LUABIND_GET_PARAMETER(1, uint, key);
  float log_threshold;
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, log_threshold,
                                 log_float::zero().log());
  log_float threshold = log_float(log_threshold);
  LMInterfaceUInt32LogFloat::ArcsIterator arcs_it =
    obj->beginNonBackoffArcs(key, threshold);
  LuaArcsIteratorUInt32Logfloat *lua_arcs_it = new
    LuaArcsIteratorUInt32Logfloat(obj, arcs_it);
  LUABIND_RETURN(LuaArcsIteratorUInt32Logfloat, lua_arcs_it);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME HistoryBasedNgramLiraLM ngram.lira.history_based_model
//BIND_CPP_CLASS    HistoryBasedNgramLiraLM
//BIND_SUBCLASS_OF  HistoryBasedNgramLiraLM HistoryBasedLMUInt32LogFloat

//BIND_LUACLASSNAME HistoryBasedNgramLiraLMInterface ngram.lira.history_based_interface
//BIND_CPP_CLASS    HistoryBasedNgramLiraLMInterface
//BIND_SUBCLASS_OF  HistoryBasedNgramLiraLMInterface HistoryBasedLMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR HistoryBasedNgramLiraLM
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "init_word_id", "trie_vector", "lira_model",
                     (const char *)0);
  unsigned int init_word_id;
  TrieVector *trie_vector;
  NgramLiraModel *lira_model;
  LUABIND_GET_TABLE_PARAMETER(1, init_word_id, uint, init_word_id);
  LUABIND_GET_TABLE_PARAMETER(1, trie_vector, TrieVector, trie_vector);
  LUABIND_GET_TABLE_PARAMETER(1, lira_model, NgramLiraModel, lira_model);
  obj = new HistoryBasedNgramLiraLM(init_word_id, trie_vector, lira_model);
  LUABIND_RETURN(HistoryBasedNgramLiraLM, obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR HistoryBasedNgramLiraLMInterface
{
  LUABIND_ERROR("Use the model method get_interface");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BunchHashedNgramLiraLM ngram.lira.bunch_hashed_model
//BIND_CPP_CLASS    BunchHashedNgramLiraLM
//BIND_SUBCLASS_OF  BunchHashedNgramLiraLM BunchHashedLMUInt32LogFloat

//BIND_LUACLASSNAME BunchHashedNgramLiraLMInterface ngram.lira.bunch_hashed_interface
//BIND_CPP_CLASS    BunchHashedNgramLiraLMInterface
//BIND_SUBCLASS_OF  BunchHashedNgramLiraLMInterface BunchHashedLMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR BunchHashedNgramLiraLM
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "bunch_size", "lira_model",
                     (const char *)0);
  unsigned int bunch_size;
  NgramLiraModel *lira_model;
  LUABIND_GET_TABLE_PARAMETER(1, bunch_size, uint, bunch_size);
  LUABIND_GET_TABLE_PARAMETER(1, lira_model, NgramLiraModel, lira_model);
  obj = new BunchHashedNgramLiraLM(bunch_size, lira_model);
  LUABIND_RETURN(BunchHashedNgramLiraLM, obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR BunchHashedNgramLiraLMInterface
{
  LUABIND_ERROR("Use the model method get_interface");
}
//BIND_END
