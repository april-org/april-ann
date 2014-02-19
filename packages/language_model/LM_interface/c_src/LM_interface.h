#ifndef LANGUAGE_MODEL_INTERFACE
#define LANGUAGE_MODEL_INTERFACE

#include "symbol_scores.h"
#include "logbase.h"
#include "referenced.h"
#include "vector.h"

namespace language_models {
  
  using april_utils::vector;

  // forma lazy de devolver un Key, en el caso de modelos
  // conexionistas no tiene sentido generar un key (es caro) que luego
  // no se va a utilizar si se aplica la poda, en el caso de modelos
  // estadisticos encapsula el estado devuelto.
  // Debe tener el metodo: Key get();
  template <typename Key>
  class FutureKey {};

  // para modelos estadisticos
  template <>
  class FutureKey<unsigned int> {
    unsigned int key;
  public:
    void set(unsigned int  k) { key = k; }
    void get(unsigned int &k) { k   = key; }
  };

  /**
     prepare => en el estadistico no hace nada, y en el conexionista
     hace un forward de la red neuronal (SOLO HASTA LA PENULTIMA CAPA).
     
     get => recibe una clave y una palabra. Devuelve un Score y un
     FutureKey, siendo este ultimo un objeto que de forma lazy se
     convierte en un Key (en el estadistico son basicamente iguales).
     
     getBestProb => la mejor probabilidad que daria un get
     
     getIntialKey => recibe el indice del context cue inicial, y
     devuelve un Key que representa al estado inicial. En el
     estadistico es el estado inicial del automata, en el conexionista
     son $n-1$ context cues.
     
     getFinalKey => recibe el indice del context cue final, y devuelve
     un Key que en el caso estadistico es el estado final del automata
     (SOLO DEBE HABER UN UNICO ESTADO FINAL), y en el conexionista un
     Key con una unica palabra (el context cue final).
   */

  typedef unsigned int Word;
  
  template <typename Key, typename Score>
  class LMInterface : public Referenced {
  public:

    struct KeyScoreTuple {
      Key   key;
      Score score;
    };
    struct FKeyScoreTuple {
      FutureKey<Key> fKey;
      Score score;
    };
    struct WordScore {
      Word word;
      Score score;
    }
    struct KeyScoreBackpointerTuple {
      Key   key;
      Score score;
      Backpointer backpointer;
    };


    virtual ~LMInterface() { }

    // get is the most basic LM query method, receives the Key and the Word,
    // returns 0,1 or several results by overwritting the vector result
    // which is cleared in the method (not need to be cleared before)
    virtual void get(const Key &key, Word word, vector<FKeyScoreTuple> result) = 0;

    // TODO: add pruning techniques
    virtual void get(const Key &key, Word word, vector<FKeyScoreTuple> result, Score pruning_threshold) = 0;


    // get in buch mode
    // this method has a default implementation based on the previous get method
    virtual void get(vector<KeyScoreBackpointerTuple> input1,
		     vector<int> sgmInput1,
		     vector<WordScore> input2,
		     vector<int> sgmInput2,
		     vector<KeyScoreBackpointerTuple> output);
    // TODO: add pruning techniques
    
		     

    virtual Score getBestProb()				             = 0;
    virtual Score getBestProb(Key &k)				     = 0;
    virtual void  getInitialKey(Word initial_word, Key &k)   = 0;

    // I don't like this method, it seems to assume that there is only one final state
    virtual void  getFinalKey(Word final_word, Key &k)       = 0;
  };
  
};

#endif // LANGUAGE_MODEL_INTERFACE
