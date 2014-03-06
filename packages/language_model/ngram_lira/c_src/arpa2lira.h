#ifndef ARPA2LIRA_H
#define ARPA2LIRA_H

#include <stdint.h>
#include "hash_table.h"
#include "vector.h"
#include "referenced.h"
#include "qsort.h"

using april_utils::hash;
using april_utils::Sort;
using april_utils::vector;

/// Arpa2Lira implementa estructuras de datos necesarias para
/// convertir los Arpa en nuestro formato interno Lira
namespace arpa2lira {

  /// Una transicion en el automata tiene:
  ///   - estado destino, palabra, probabilidad
  struct Transition {
    uint32_t  state, word;
    log_float prob;
    /// oprdena por ID de las palabras
    bool operator<(const Transition &other) const {
      return word < other.word;
    }
  };
  
  /// tipos utiles
  typedef vector<Transition> TransitionsType;
  typedef hash<uint32_t,TransitionsType> HashType;

  /// permite tener los vectores de transiciones en Lua
  struct VectorReferenced : public Referenced {
    TransitionsType &v;
    VectorReferenced(TransitionsType &v) : v(v) { }
    void sortByWordId() {
      Sort(v.begin(), v.size());
    }
  };

  /// Iterador de transiciones: util para iterar sobre la tabla hash
  /// en Lua
  class TransitionsIterator : public Referenced {
    HashType::iterator it;
  public:
    TransitionsIterator(HashType::iterator it) : it(it) {
    }
    void next() { it++; }
    TransitionsType &getTransitions() { return it->second; }
    uint32_t getState() { return it->first; }
    bool notEqual(TransitionsIterator *other) const {
      return other->it != it;
    }
    /// OJO: idx empieza en 0
    void set(uint32_t idx, uint32_t state, uint32_t word, log_float prob) {
      it->second[idx].state = state;
      it->second[idx].word  = word;
      it->second[idx].prob  = prob;
    }
  };
  
  /// Clase que permite asociar a un estado un vector de transiciones
  class State2Transitions : public Referenced {
    /// tabla Hash
    HashType data;
  public:
    State2Transitions() {
    }
    ~State2Transitions() {
    }
    bool exists(uint32_t st) {
      return data.find(st) != 0;
    }
    void create(uint32_t st) {
      TransitionsType &t = data[st];
      t.clear();
    }
    void insert(uint32_t  orig_state,
		uint32_t  dest_state,
		uint32_t  word,
		log_float prob) {
      Transition t;
      t.state = dest_state;
      t.word  = word;
      t.prob  = prob;
      data[orig_state].push_back(t);
    }
    void erase(uint32_t st) {
      data.erase(st);
    }
    TransitionsIterator begin() {
      return TransitionsIterator(data.begin());
    }
    TransitionsIterator end() {
      return TransitionsIterator(data.end());
    }
    TransitionsType &getTransitions(uint32_t st) {
      return data[st];
    }
  };

}

#endif // ARPA2LIRA_H
