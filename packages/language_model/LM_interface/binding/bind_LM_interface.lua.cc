//BIND_HEADER_C
#include "bind_dataset.h"
#include "bind_LM_interface.h"
#include "LM_interface.h"
//BIND_END

//BIND_HEADER_H
#include "LM_interface.h"
using namespace LanguageModels;

class QueryResultUInt32LogFloat : public Referenced {
  const vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> &result;
public:
  QueryResultUInt32LogFloat(const vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> &result) :
    result(result) {}
  size_t size() const { return result.size(); }
  const LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple &get(unsigned int i) const {
    return result[i];
  }
};

class GetResultUInt32LogFloat : public Referenced {
  vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> result;
public:
  GetResultUInt32LogFloat() {}
  vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> &getVector() {
    return result;
  }
  size_t size() const { return result.size(); }
  const LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple &get(unsigned int i) const {
    return result[i];
  }
};

//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME QueryResultUInt32LogFloat language_models.query_result
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

//BIND_LUACLASSNAME GetResultUInt32LogFloat language_models.get_result
//BIND_CPP_CLASS    GetResultUInt32LogFloat language_models.get_result

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
  bool clear_result = true;
  GetResultUInt32LogFloat *result = 0;
  LUABIND_GET_PARAMETER(1, uint, key);
  LUABIND_GET_PARAMETER(2, uint, word);
  if (lua_istable(L,3)) {
    check_table_fields(L, 3, "threshold", "id_key", "id_word", "result", "clear_result",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, threshold, float, log_threshold,
					 log_float::zero());
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, id_key, int, burden_id_key, -1);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, id_word, int, burden_id_word, -1);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, result, GetResultUInt32LogFloat,
					 result, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(3, clear_result, bool, clear_result, true);
    threshold = log_float(log_threshold);
  }
  if (result == 0) result = new GetResultUInt32LogFloat;
  if (clear_result) result->getVector():clear();
  obj->get(key, word, LMInterfaceUInt32LogFloat::Burden(burden_id_key,
							burden_id_word),
	   result->getVector(), threshold);
  LUABIND_RETURN(GetResultUInt32LogFloat, result);
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
					 log_float::zero());
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
  const vector<LMInterfaceUInt32LogFloat::KeyScoreBurdenTuple> &vresult = obj->getQueries();
  QueryResultUInt32LogFloat *result = new QueryResultUInt32LogFloat(vresult);
  LUABIND_RETURN(QueryResultUInt32LogFloat, result);
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_initial_key
{
  uint32_t key;
  obj->getInitialKey(key);
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
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, log_threshold, log_float::zero());
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
