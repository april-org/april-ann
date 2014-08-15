//BIND_HEADER_H
#include "arpa2lira.h"
using LanguageModels::arpa2lira::Transition;
using LanguageModels::arpa2lira::TransitionsType;
using LanguageModels::arpa2lira::TransitionsIterator;
using LanguageModels::arpa2lira::State2Transitions;
using LanguageModels::arpa2lira::VectorReferenced;
//BIND_END

//BIND_HEADER_C
using april_utils::vector;
using april_utils::log_float;
//BIND_END

////////////////////////////////////

//BIND_LUACLASSNAME VectorReferenced ngram.lira.arpa2lira.VectorReferenced
//BIND_CPP_CLASS    VectorReferenced

//BIND_CONSTRUCTOR VectorReferenced
{
  LUABIND_ERROR("FORBIDDEN\n");
  exit(128);
}
//BIND_END

//BIND_METHOD VectorReferenced size
{
  LUABIND_RETURN(uint, obj->v.size());
}
//BIND_END

//BIND_METHOD VectorReferenced get
{
  unsigned int idx;
  LUABIND_GET_PARAMETER(1, uint, idx);
  LUABIND_RETURN(uint,  obj->v[idx-1].state);
  LUABIND_RETURN(uint,  obj->v[idx-1].word);
  LUABIND_RETURN(float, obj->v[idx-1].prob.log());
}
//BIND_END

//BIND_METHOD VectorReferenced set
{
  unsigned int	idx;
  uint32_t	state,word;
  float		lprob;
  LUABIND_GET_PARAMETER(1, uint,  idx);
  LUABIND_GET_PARAMETER(2, uint,  state);
  LUABIND_GET_PARAMETER(3, uint,  word);
  LUABIND_GET_PARAMETER(4, float, lprob);
  obj->v[idx-1].state = state;
  obj->v[idx-1].word  = word;
  obj->v[idx-1].prob  = log_float(lprob);
}
//BIND_END

//BIND_METHOD VectorReferenced sortByWordId
{
  obj->sortByWordId();
}
//BIND_END

////////////////////////////////////

//BIND_LUACLASSNAME TransitionsIterator ngram.lira.arpa2lira.TransitionsIterator
//BIND_CPP_CLASS    TransitionsIterator

//BIND_CONSTRUCTOR TransitionsIterator
{
  LUABIND_ERROR("FORBIDDEN\n");
  exit(128);
}
//BIND_END

//BIND_METHOD TransitionsIterator getState
{
  LUABIND_RETURN(uint, obj->getState());
}
//BIND_END

//BIND_METHOD TransitionsIterator getTransitions
{
  VectorReferenced *v = new VectorReferenced(obj->getTransitions());
  LUABIND_RETURN(VectorReferenced, v);
}
//BIND_END

//BIND_METHOD TransitionsIterator next
{
  obj->next();
}
//BIND_END

//BIND_METHOD TransitionsIterator notEqual
{
  TransitionsIterator *other;
  LUABIND_GET_PARAMETER(1, TransitionsIterator, other);
  LUABIND_RETURN(bool, obj->notEqual(other));
}
//BIND_END

////////////////////////////////////

//BIND_LUACLASSNAME State2Transitions ngram.lira.arpa2lira.State2Transitions
//BIND_CPP_CLASS    State2Transitions

//BIND_CONSTRUCTOR State2Transitions
{
  obj = new State2Transitions();
  LUABIND_RETURN(State2Transitions, obj);
}
//BIND_END

//BIND_METHOD State2Transitions exists
{
  unsigned int st;
  LUABIND_GET_PARAMETER(1, uint, st);
  bool ret = obj->exists(st);
  LUABIND_RETURN(bool, ret);
}
//BIND_END

//BIND_METHOD State2Transitions create
{
  unsigned int st;
  LUABIND_GET_PARAMETER(1, uint, st);
  obj->create(st);
}
//BIND_END

//BIND_METHOD State2Transitions insert
{
  unsigned int orig_st, dest_st, word;
  float lprob;
  LUABIND_GET_PARAMETER(1, uint, orig_st);
  LUABIND_GET_PARAMETER(2, uint, dest_st);
  LUABIND_GET_PARAMETER(3, uint, word);
  LUABIND_GET_PARAMETER(4, float, lprob);
  obj->insert(orig_st, dest_st, word, log_float(lprob));
}
//BIND_END

//BIND_METHOD State2Transitions beginIt
{
  TransitionsIterator it = obj->begin();
  TransitionsIterator *ptr_it = new TransitionsIterator(it);
  LUABIND_RETURN(TransitionsIterator, ptr_it);
}
//BIND_END

//BIND_METHOD State2Transitions endIt
{
  TransitionsIterator it = obj->end();
  TransitionsIterator *ptr_it = new TransitionsIterator(it);
  LUABIND_RETURN(TransitionsIterator, ptr_it);
}
//BIND_END

//BIND_METHOD State2Transitions erase
{
  unsigned int st;
  LUABIND_GET_PARAMETER(1, uint, st);
  obj->erase(st);
}
//BIND_END

//BIND_METHOD State2Transitions getTransitions
{
  unsigned int st;
  LUABIND_GET_PARAMETER(1, uint, st);
  TransitionsType &t = obj->getTransitions(st);
  VectorReferenced *v = new VectorReferenced(t);
  LUABIND_RETURN(VectorReferenced, v);
}
//BIND_END
