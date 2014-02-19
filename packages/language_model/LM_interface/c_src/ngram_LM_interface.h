#ifndef LANGUAGE_MODEL_INTERFACE
#define LANGUAGE_MODEL_INTERFACE

#include "LM_interface.h"

namespace language_models {
  
  template <typename Key, typename Score>
  class ngramLMInterface : public LMInterface {
  public:
    
    virtual ~LMInterface() { }
    virtual void  getZeroGramKey(unsigned int initial_word, Key &k)  = 0;
    virtual unsigned int getLMOrder()                                = 0;
  };
  
};

#endif // LANGUAGE_MODEL_INTERFACE
