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
//BIND_HEADER_C
#include "bind_dataset.h"
#include "bind_function_interface.h"
#include "bind_LM_interface.h"
#include "LM_interface.h"

using namespace AprilUtils;
using namespace Basics;
using namespace Functions;
//BIND_END

//BIND_HEADER_H
#include "LM_interface.h"
#include "history_based_LM.h"
#include "feature_based_LM.h"
#include "bunch_hashed_LM.h"
#include "skip_function.h"
using namespace LanguageModels;
using namespace LanguageModels::QueryFilters;

class LuaArcsIteratorUInt32Logfloat : public Referenced {
  AprilUtils::SharedPtr<LMInterfaceUInt32LogFloat> lmi;
  uint32_t key;
  LMInterfaceUInt32LogFloat::ArcsIterator it;
public:
  LuaArcsIteratorUInt32Logfloat(LMInterfaceUInt32LogFloat *lmi,
                                LMInterfaceUInt32LogFloat::ArcsIterator &it) :
    Referenced(),
    lmi(lmi), it(it) {
  }
  uint32_t get() {
    return *it;
  }
  void next() {
    ++it;
  }
  bool isEnd() {
    return it.isEnd();
  }
};

class QueryResultUInt32LogFloat : public Referenced {
  LMInterfaceUInt32LogFloat *lm_interface;
  const AprilUtils::vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> &result;
public:
  QueryResultUInt32LogFloat(LMInterfaceUInt32LogFloat *lm_interface) :
    lm_interface(lm_interface),
    result(lm_interface->getQueries()) {
    IncRef(lm_interface);
  }
  ~QueryResultUInt32LogFloat() {
    DecRef(lm_interface);
  }
  size_t size() const { return result.size(); }
  const LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple &get(unsigned int i) const {
    return result[i];
  }
};

class GetResultUInt32LogFloat : public Referenced {
  AprilUtils::vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> result;
public:
  GetResultUInt32LogFloat() {}
  AprilUtils::vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> &getVector() {
    return result;
  }
  size_t size() const { return result.size(); }
  const LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple &get(unsigned int i) const {
    return result[i];
  }
  void clear() {
    result.clear();
  }
};

class NextKeysResultUInt32 : public Referenced {
  AprilUtils::vector<uint32_t> result;
public:
  NextKeysResultUInt32() {}
  AprilUtils::vector<uint32_t> &getVector() {
    return result;
  }
  size_t size() const { return result.size(); }
  const uint32_t &get(unsigned int i) const {
    return result[i];
  }
  void clear() {
    result.clear();
  }
};

//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME LuaArcsIteratorUInt32Logfloat language_models.__basic_arcs_iterator__
//BIND_CPP_CLASS LuaArcsIteratorUInt32Logfloat

//BIND_CONSTRUCTOR LuaArcsIteratorUInt32Logfloat
{
  LUABIND_ERROR("FORBIDDEN!!! call the corresponding method in language model");
}
//BIND_END

//BIND_METHOD LuaArcsIteratorUInt32Logfloat get
{
  LUABIND_RETURN(uint, obj->get());
}
//BIND_END

//BIND_METHOD LuaArcsIteratorUInt32Logfloat next
{
  obj->next();
}
//BIND_END

//BIND_METHOD LuaArcsIteratorUInt32Logfloat is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME QueryResultUInt32LogFloat language_models.__query_result__
//BIND_CPP_CLASS    QueryResultUInt32LogFloat

//BIND_CONSTRUCTOR QueryResultUInt32LogFloat
{
  LUABIND_ERROR("FORBIDDEN!!! call method get_queries of a language model");
}
//BIND_END

//BIND_METHOD QueryResultUInt32LogFloat get
{
  unsigned int i;
  LUABIND_GET_PARAMETER(1, uint, i);
  LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple tuple = obj->get(i-1);
  LUABIND_RETURN(uint, tuple.key_score.key);
  LUABIND_RETURN(float, tuple.key_score.score.log());
  LUABIND_RETURN(int, tuple.burden.id_key);
  LUABIND_RETURN(int, tuple.burden.id_word);
}
//BIND_END

//BIND_METHOD QueryResultUInt32LogFloat to_table
{
  lua_newtable(L);
  for (unsigned int i=0; i<obj->size(); ++i) {
    LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple tuple = obj->get(i);
    lua_pushnumber(L, i+1);
    lua_newtable(L);
    //
    lua_pushnumber(L, 1);
    lua_pushuint(L, tuple.key_score.key);
    lua_settable(L, -3);
    //
    lua_pushnumber(L, 2);
    lua_pushfloat(L, tuple.key_score.score.log());
    lua_settable(L, -3);
    //
    lua_pushnumber(L, 3);
    lua_pushint(L, tuple.burden.id_key);
    lua_settable(L, -3);
    //
    lua_pushnumber(L, 4);
    lua_pushint(L, tuple.burden.id_word);
    lua_settable(L, -3);
    //
    lua_settable(L, -3);
  }
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD QueryResultUInt32LogFloat size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME GetResultUInt32LogFloat language_models.__get_result__
//BIND_CPP_CLASS    GetResultUInt32LogFloat

//BIND_CONSTRUCTOR GetResultUInt32LogFloat
{
  LUABIND_ERROR("FORBIDDEN!!! call method get of a language model");
}
//BIND_END

//BIND_METHOD GetResultUInt32LogFloat get
{
  unsigned int i;
  LUABIND_GET_PARAMETER(1, uint, i);
  LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple tuple = obj->get(i-1);
  LUABIND_RETURN(uint, tuple.key_score.key);
  LUABIND_RETURN(float, tuple.key_score.score.log());
  LUABIND_RETURN(int, tuple.burden.id_key);
  LUABIND_RETURN(int, tuple.burden.id_word);
}
//BIND_END

//BIND_METHOD GetResultUInt32LogFloat clear
{
  obj->clear();
  LUABIND_RETURN(GetResultUInt32LogFloat, obj);
}
//BIND_END

//BIND_METHOD GetResultUInt32LogFloat size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD GetResultUInt32LogFloat to_table
{
  lua_newtable(L);
  for (unsigned int i=0; i<obj->size(); ++i) {
    LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple tuple = obj->get(i);
    lua_pushnumber(L, i+1);
    lua_newtable(L);
    //
    lua_pushnumber(L, 1);
    lua_pushuint(L, tuple.key_score.key);
    lua_settable(L, -3);
    //
    lua_pushnumber(L, 2);
    lua_pushfloat(L, tuple.key_score.score.log());
    lua_settable(L, -3);
    //
    lua_pushnumber(L, 3);
    lua_pushint(L, tuple.burden.id_key);
    lua_settable(L, -3);
    //
    lua_pushnumber(L, 4);
    lua_pushint(L, tuple.burden.id_word);
    lua_settable(L, -3);
    //
    lua_settable(L, -3);
  }
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME NextKeysResultUInt32 language_models.__next_keys_result__
//BIND_CPP_CLASS    NextKeysResultUInt32

//BIND_CONSTRUCTOR NextKeysResultUInt32
{
  LUABIND_ERROR("FORBIDDEN!!! call method get of a language model");
}
//BIND_END

//BIND_METHOD NextKeysResultUInt32 get
{
  unsigned int i;
  LUABIND_GET_PARAMETER(1, uint, i);
  uint32_t key = obj->get(i-1);
  LUABIND_RETURN(uint, key);
}
//BIND_END

//BIND_METHOD NextKeysResultUInt32 clear
{
  obj->clear();
  LUABIND_RETURN(NextKeysResultUInt32, obj);
}
//BIND_END

//BIND_METHOD NextKeysResultUInt32 size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD NextKeysResultUInt32 to_table
{
  lua_newtable(L);
  for (unsigned int i=0; i<obj->size(); ++i) {
    uint32_t key = obj->get(i);
    lua_pushnumber(L, i+1);
    lua_pushuint(L, key);
    lua_settable(L, -3);
  }
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME LMModelUInt32LogFloat language_models.model
//BIND_CPP_CLASS    LMModelUInt32LogFloat

//BIND_CONSTRUCTOR LMModelUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD LMModelUInt32LogFloat get_interface
{
  LUABIND_RETURN(LMInterfaceUInt32LogFloat, obj->getInterface());
}
//BIND_END

//BIND_METHOD LMModelUInt32LogFloat is_deterministic
{
  LUABIND_RETURN(boolean, obj->isDeterministic());
}
//BIND_END

//BIND_METHOD LMModelUInt32LogFloat ngram_order
{
  LUABIND_RETURN(int, obj->ngramOrder());
}
//BIND_END

//BIND_METHOD LMModelUInt32LogFloat require_history_manager
{
  LUABIND_RETURN(boolean, obj->requireHistoryManager());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME LMInterfaceUInt32LogFloat language_models.interface
//BIND_CPP_CLASS    LMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR LMInterfaceUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get
{
  uint32_t key, word;
  int burden_id_key=-1, burden_id_word=-1;
  float log_threshold;
  log_float threshold = log_float::zero();
  GetResultUInt32LogFloat *result = 0;
  LUABIND_GET_PARAMETER(1, uint, key);
  LUABIND_GET_PARAMETER(2, uint, word);
  if (lua_istable(L,3)) {
    check_table_fields(L, 3, "threshold", "id_key", "id_word", "result",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, threshold, float, log_threshold,
					 log_float::zero().log());
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, id_key, int, burden_id_key, -1);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, id_word, int, burden_id_word, -1);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, result, GetResultUInt32LogFloat,
					 result, 0);
    threshold = log_float(log_threshold);
  }
  if (result == 0) result = new GetResultUInt32LogFloat;
  obj->get(key, word, LMInterfaceUInt32LogFloat::Burden(burden_id_key,
							burden_id_word),
	   result->getVector(), threshold);
  LUABIND_RETURN(GetResultUInt32LogFloat, result);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat next_keys
{
  uint32_t key, word;
  NextKeysResultUInt32 *result = 0;
  LUABIND_GET_PARAMETER(1, uint, key);
  LUABIND_GET_PARAMETER(2, uint, word);
  if (lua_istable(L,3)) {
    check_table_fields(L, 3, "result",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, result, NextKeysResultUInt32,
					 result, 0);
  }
  if (result == 0) result = new NextKeysResultUInt32;
  obj->getNextKeys(key, word, result->getVector());
  LUABIND_RETURN(NextKeysResultUInt32, result);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat clear_queries
{
  obj->clearQueries();
  LUABIND_RETURN(LMInterfaceUInt32LogFloat, obj);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat insert_query
{
  uint32_t key, word;
  float log_threshold;
  log_float threshold = log_float::zero();
  int burden_id_key=-1, burden_id_word=-1;
  LUABIND_GET_PARAMETER(1, uint, key);
  LUABIND_GET_PARAMETER(2, uint, word);
  if (lua_istable(L,3)) {
    check_table_fields(L, 3, "threshold", "id_key", "id_word", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, threshold, float, log_threshold,
					 log_float::zero().log());
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, id_key, int, burden_id_key, -1);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, id_word, int, burden_id_word, -1);
    threshold = log_float(log_threshold);
  }
  obj->insertQuery(key, word,
		   LMInterfaceUInt32LogFloat::Burden(burden_id_key,
						     burden_id_word),
		   threshold);
  LUABIND_RETURN(LMInterfaceUInt32LogFloat, obj);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_queries
{
  QueryResultUInt32LogFloat *result = new QueryResultUInt32LogFloat(obj);
  LUABIND_RETURN(QueryResultUInt32LogFloat, result);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_initial_key
{
  uint32_t key = obj->getInitialKey();
  LUABIND_RETURN(uint, key);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_final_score
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  uint32_t key;
  LUABIND_GET_PARAMETER(1, uint, key);
  float log_threshold;
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, log_threshold,
                                 log_float::zero().log());
  log_float threshold = log_float(log_threshold);
  LUABIND_RETURN(float, obj->getFinalScore(key, threshold).log());
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_zero_key
{
  uint32_t key;
  if (!obj->getZeroKey(key))
    LUABIND_ERROR("Impossible to get zero key");
  LUABIND_RETURN(uint, key);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_best_prob
{
  uint32_t key;
  log_float score;
  if (lua_isnil(L,1)) {
    score = obj->getBestProb();
  }
  else {
    LUABIND_GET_PARAMETER(1, uint, key);
    score = obj->getBestProb(key);
  }
  LUABIND_RETURN(float, score.log());
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME HistoryBasedLMUInt32LogFloat language_models.history_based_model
//BIND_CPP_CLASS HistoryBasedLMUInt32LogFloat
//BIND_SUBCLASS_OF HistoryBasedLMUInt32LogFloat LMModelUInt32LogFloat

//BIND_CONSTRUCTOR HistoryBasedLMUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME HistoryBasedLMInterfaceUInt32LogFloat language_models.history_based_interface
//BIND_CPP_CLASS HistoryBasedLMInterfaceUInt32LogFloat
//BIND_SUBCLASS_OF HistoryBasedLMInterfaceUInt32LogFloat LMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR HistoryBasedLMInterfaceUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BunchHashedLMUInt32LogFloat language_models.bunch_hashed_model
//BIND_CPP_CLASS BunchHashedLMUInt32LogFloat
//BIND_SUBCLASS_OF BunchHashedLMUInt32LogFloat LMModelUInt32LogFloat

//BIND_CONSTRUCTOR BunchHashedLMUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BunchHashedLMInterfaceUInt32LogFloat language_models.bunch_hashed_interface
//BIND_CPP_CLASS BunchHashedLMInterfaceUInt32LogFloat
//BIND_SUBCLASS_OF BunchHashedLMInterfaceUInt32LogFloat LMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR BunchHashedLMInterfaceUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FeatureBasedLMUInt32LogFloat language_models.feature_based_model
//BIND_CPP_CLASS FeatureBasedLMUInt32LogFloat
//BIND_SUBCLASS_OF FeatureBasedLMUInt32LogFloat LMModelUInt32LogFloat

//BIND_CONSTRUCTOR FeatureBasedLMUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FeatureBasedLMInterfaceUInt32LogFloat language_models.feature_based_interface
//BIND_CPP_CLASS FeatureBasedLMInterfaceUInt32LogFloat
//BIND_SUBCLASS_OF FeatureBasedLMInterfaceUInt32LogFloat LMInterfaceUInt32LogFloat

//BIND_CONSTRUCTOR FeatureBasedLMInterfaceUInt32LogFloat
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FunctionInterface functions

//BIND_LUACLASSNAME DiceSkipFunction functions.dice_skip
//BIND_CPP_CLASS DiceSkipFunction
//BIND_SUBCLASS_OF DiceSkipFunction FunctionInterface

//BIND_CONSTRUCTOR DiceSkipFunction
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "dice", "random", "mask_value");
  Dice *dice;
  MTRand *random;
  uint32_t mask_value;
  LUABIND_GET_TABLE_PARAMETER(1, dice, Dice, dice);
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, random);
  LUABIND_GET_TABLE_PARAMETER(1, mask_value, uint, mask_value);
  obj = new DiceSkipFunction(dice, random, mask_value);
  LUABIND_RETURN(DiceSkipFunction, obj);
}
//BIND_END
