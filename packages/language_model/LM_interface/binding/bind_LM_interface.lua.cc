//BIND_HEADER_C
#include "bind_dataset.h"
#include "bind_LM_interface.h"
#include "LM_interface.h"
//BIND_END

//BIND_HEADER_H
#include "LM_interface.h"
using namespace LanguageModels;

class QueryResult : public Referenced {
  const vector<KeyScoreBurdenTuple> &result;
public:
  QueryResult(const vector<KeyScoreBurdenTuple> &result) : result(result) {}
  size_t size() const { return result.size(); }
  const KeyScoreBurdenTuple &get(int i) const { return result[i]; }
};

//BIND_END

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

//BIND_METHOD LMInterfaceUInt32LogFloat get_model
{
  LUABIND_RETURN(LMModelUInt32LogFloat, obj->getLMModel());
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get
{
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat clear_queries
{
  
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat insert_query
{
}
//BIND_END

//BIND_METHOD LMInterfaceUInt32LogFloat get_queries
{
}
//BIND_END

