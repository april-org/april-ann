/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef TRIE_VECTOR_H
#define TRIE_VECTOR_H

extern "C" {
#include <stdint.h>
}
#include "referenced.h"

/*

Estructura de datos para representar tries de palabras. Las palabras
son uint32 con valores entre 0 y 2**24-2 (la 2**24-1 se reserva)

NO SOPORTA REHASHING, pues cada nodo se identifica con su indice
en el vector de open addressing.

Cada nodo contiene el indice de su padre y la ultima palabra
introducida. Para acceder a un nodo hijo se hace hashing con el propio
nodo padre y la palabra introducida.

Utiliza timestamp de talla 1 byte, lo que supone recorrer la tabla
cada 255 llamadas a "clear()"

El valor de timestamp 0 se reserva para nodos "persistentes" que no se
borran con la operacion "clear".

Se presupone que todas las inserciones persistentes se realizan al
principio, con lo que no hace falta comprobar que todos los
antecesores de un nodo persistente tambien lo son.

*/

namespace april_utils {

  class TrieVector : public Referenced {
    unsigned int max_allowed_size;
    
    static const uint32_t NoWord = (1<<24)-1;

    struct TrieNode {
      union {
	struct {
	  uint32_t parent;    // indice en vector
	  uint32_t wordStamp; // word<<8|stamp
	};
	uint64_t rawValue;
      };
      TrieNode(uint32_t parent=0, uint32_t wordStamp=0) :
	parent(parent), wordStamp(wordStamp) {}
      bool operator!= (const TrieNode &other) const {
	return rawValue != other.rawValue;
      }
      bool operator== (const TrieNode &other) const {
	return rawValue == other.rawValue;
      }
    };

    unsigned int vectorSize; // talla del vector, potencia de 2
    unsigned int mask; // vectorSize-1
    unsigned int size; // elementos insertados
    unsigned int stamp; // actual
    TrieNode *data; // el vector
  public:
    TrieVector(int logSize=20);
    ~TrieVector();
    unsigned int getSize() const { return size; }
    uint32_t getParent(uint32_t node) const { return data[node].parent; }
    uint32_t getWord(uint32_t node)   const { return data[node].wordStamp>>8; }

    // busqueda sin insertar
    bool   hasChild (uint32_t node, uint32_t word, uint32_t &destnode);
    // busqueda insertando
    uint32_t getChild (uint32_t node, uint32_t word);
    uint32_t getPersistentChild (uint32_t node, uint32_t word);

    // busqueda sin insertar
    bool   hasSequence(const uint32_t *sequence, int length, uint32_t &destnode);
    // busqueda insertando
    uint32_t searchSequence(const uint32_t *sequence, int length);
    uint32_t searchPersistentSequence(const uint32_t *sequence, int length);

    /// devuelve la longitud, -2 si no existe, -1 si no cabe
    int    getSequence(uint32_t node, uint32_t *sequence, int maxLength);

    uint32_t rootNode() const { return 0; }
    void clear();
  }; // class TrieVector

} // namespace

#endif // TRIE_VECTOR_H
