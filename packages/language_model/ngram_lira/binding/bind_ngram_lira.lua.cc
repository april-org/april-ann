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
//BIND_END

//BIND_HEADER_H
#include "ngram_lira.h"
using namespace LanguageModels;
//BIND_END

//BIND_LUACLASSNAME NgramLiraModel ngram.lira.model
//BIND_CPP_CLASS    NgramLiraModel
//BIND_SUBCLASS_OF  NgramLiraModel LMModel

//BIND_LUACLASSNAME NgramLiraInterface ngram.lira.interface
//BIND_CPP_CLASS    NgramLiraInterface
//BIND_SUBCLASS_OF  NgramLiraModel LMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR NgramLiraModel
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, 
		     "binary", // boolean
		     "command", // open file command, i.e. "zcat blah.gz"
		     "filename",
		     "vocabulary",
		     "fan_out_threshold",
		     // sirve para permitir que el diccionario que te
		     // pasan sea mas grande que el que hay en el
		     // modelo lira
		     "ignore_extra_words_in_dictionary",
		     0); // 0 para terminar el check_table_fields
  //
  bool ignore_extra_words_in_dictionary = false;
  bool binary = false;
  const char *filename;
  const char *command;
  NgramLiraModel *obj=0;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, binary,
				       bool,
				       binary, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, filename, string, filename, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, ignore_extra_words_in_dictionary,
				       bool,
				       ignore_extra_words_in_dictionary, false);

  lua_getfield(L,1,"vocabulary");
  if (lua_isnil(L,-1)) {
    LUABIND_ERROR("error ngram.lira.model constructor requires vocabulary table");
  }
  int vocabulary_size;
  LUABIND_TABLE_GETN(-1, vocabulary_size);
  const char **vocabulary_vector = new const char *[vocabulary_size];
  LUABIND_TABLE_TO_VECTOR(-1, string, vocabulary_vector, vocabulary_size);

  if (binary) {

    obj=new NgramLiraModel(filename,
			   (unsigned int)vocabulary_size,
			   vocabulary_vector,
			   ignore_extra_words_in_dictionary);
  } else {
    FILE *fd;
    int fan_out_threshold;
    LUABIND_GET_TABLE_PARAMETER(1, fan_out_threshold, int, fan_out_threshold);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, command, string, command, 0);
    if (command) {
      fd = popen(command,"r");
      if (fd == 0) {
	LUABIND_FERROR1("unable to popen %s\n",command);
      }
    }
    else {
      fd = fopen(filename, "r");
      if (fd == 0) {
	LUABIND_FERROR1("unable to fopen %s\n",filename);
      }
    }
    obj=new NgramLiraModel(fd,
			   (unsigned int)vocabulary_size,
			   vocabulary_vector,
			   fan_out_threshold,
			   ignore_extra_words_in_dictionary);
    if (command) pclose(fd);
    else fclose(fd);
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
		     0); // 0 para terminar el check_table_fields
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
  
  obj->save_binary(filename,
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
		     0); // 0 para terminar el check_table_fields
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
  LAUBIND_ERROR("Use the model method get_interface");
}
//BIND_END
