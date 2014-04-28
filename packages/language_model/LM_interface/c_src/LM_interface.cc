#include "LM_interface.h"

template<> class LMInterface<uint32_t,log_float>;
template<> class LMModel<uint32_t,log_float>;

template<> class HistoryBasedLMInterface<uint32_t,log_float>;
template<> class HistoryBasedLM<uint32_t,log_float>;

template<> class BunchHashedLMInterface<uint32_t,log_float>;
template<> class BunchHashedLM<uint32_t,log_float>;
