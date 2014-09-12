//BIND_HEADER_C
//BIND_END

//BIND_HEADER_H
#include <cstring>
#include "lexclass.h"
using namespace AprilUtils;
//BIND_END

//BIND_LUACLASSNAME LexClass _internal_lexclass_
//BIND_CPP_CLASS    LexClass

//BIND_CONSTRUCTOR LexClass
{
  LexClass *obj = new LexClass();
  LUABIND_RETURN(LexClass, obj);
}
//BIND_END

//BIND_DESTRUCTOR LexClass
{
}
//BIND_END

//BIND_METHOD LexClass addPair
{
  const char	*lua_word, *lua_outsym;
  char          *word, *outsym;
  log_float	 lprob;
  float		 prob;
  
  check_table_fields(L, 1, "word", "outsym", "prob", 0);
  
  LUABIND_GET_TABLE_PARAMETER(1, word, string, lua_word);
  LUABIND_GET_TABLE_PARAMETER(1, outsym, string, lua_outsym);
  LUABIND_GET_TABLE_PARAMETER(1, prob, float, prob);
  lprob	 = log_float(prob);
  word	 = new char[strlen(lua_word)+1];
  strcpy(word, lua_word);
  outsym = new char[strlen(lua_outsym)+1];
  strcpy(outsym, lua_outsym);
  
  uint32_t ret = obj->addPair(word, outsym, prob);
  LUABIND_RETURN(uint, ret);
}
//BIND_END

//BIND_METHOD LexClass size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD LexClass getWordFromWordId
{
  uint32_t wid;
  LUABIND_GET_PARAMETER(1, uint, wid);
  const char *w = obj->getWordFromWordId(wid);
  LUABIND_RETURN(string, w);
}
//BIND_END

//BIND_METHOD LexClass getWordFromPairId
{
  uint32_t pid;
  LUABIND_GET_PARAMETER(1, uint, pid);
  const char *w = obj->getWordFromPairId(pid);
  LUABIND_RETURN(string, w);
}
//BIND_END

//BIND_METHOD LexClass getOutSymFromOutSymId
{
  uint32_t oid;
  LUABIND_GET_PARAMETER(1, uint, oid);
  const char *o = obj->getOutSymFromOutSymId(oid);
  LUABIND_RETURN(string, o);
}
//BIND_END

//BIND_METHOD LexClass getOutSymFromPairId
{
  uint32_t pid;
  LUABIND_GET_PARAMETER(1, uint, pid);
  const char *o = obj->getOutSymFromPairId(pid);
  LUABIND_RETURN(string, o);
}
//BIND_END

//BIND_METHOD LexClass getWordId
{
  uint32_t wid;
  const char *w;
  LUABIND_GET_PARAMETER(1, string, w);
  if (obj->getWordId(w, wid)) LUABIND_RETURN(uint, wid);
  else LUABIND_RETURN_NIL();
}
//BIND_END

//BIND_METHOD LexClass getOutSymId
{
  uint32_t oid;
  const char *o;
  LUABIND_GET_PARAMETER(1, string, o);
  if (obj->getOutSymId(o, oid)) LUABIND_RETURN(uint, oid);
  else LUABIND_RETURN_NIL();
}
//BIND_END

//BIND_METHOD LexClass getPairData
{
  uint32_t pair_id;
  LUABIND_GET_PARAMETER(1, uint, pair_id);
  LexClass::data_t d = obj->getPairData(pair_id);
  
  lua_newtable(L);
  LUABIND_RETURN_FROM_STACK(-1);
  
  lua_pushstring(L, "word");
  lua_pushnumber(L, d.word);
  lua_settable(L, -3);

  lua_pushstring(L, "outsym");
  lua_pushnumber(L, d.outsym);
  lua_settable(L, -3);

  lua_pushstring(L, "prob");
  lua_pushnumber(L, d.prob.log());
  lua_settable(L, -3);
}
//BIND_END

//BIND_METHOD LexClass wordTblSize
{
  LUABIND_RETURN(uint, obj->wordTblSize());
}
//BIND_END

//BIND_METHOD LexClass outsymTblSize
{
  LUABIND_RETURN(uint, obj->outsymTblSize());
}
//BIND_END
