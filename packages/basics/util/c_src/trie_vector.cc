/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include <cstring> // memset
#include "trie_vector.h"
#include "error_print.h"
#include "swap.h"

namespace april_utils {

  TrieVector::TrieVector(int logSize) {
    vectorSize       = 1<<logSize;
    mask             = vectorSize-1;
    max_allowed_size = vectorSize*0.8;
    size             = 0;
    data             = new TrieNode[vectorSize]; // alineado al menos a uint64 ;)
    stamp            = 2;
    for (unsigned int i=0; i<vectorSize; ++i)
      data[i].wordStamp = 1; // stamp 1 para indicar que estan vacios
    // el nodo raiz es persistente:
    data[0].wordStamp = NoWord<<8;
  }

  void TrieVector::clear() {
    size = 0;
    stamp++;
    if (stamp == 256) {
      for (unsigned int i=0; i<vectorSize; ++i)
	if ((data[i].wordStamp & 255) != 0)
	  data[i].wordStamp = 1;
      stamp = 2;
    }
  }

  TrieVector::~TrieVector() {
    delete[] data;
  }

  // busca un nodo que puede ser persistente o no:
  uint32_t TrieVector::getChild (uint32_t node, uint32_t word) {
     static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    TrieNode searchedP(node,word<<8);
    TrieNode searchedE(node,(word<<8)|stamp);
    unsigned int index     = (node*cte_hash ^ word) & mask;
    unsigned int increment = word | 1;
    for (;;) {
      unsigned int stmp = data[index].wordStamp & 255;
      if (stmp == 0) { // es un nodo persistente
	if (data[index] == searchedP) return index;
      } else if (stmp == stamp) {
	if (data[index] == searchedE) return index;
      } else { // es un nodo vacio, insertamos
	data[index] = searchedE;
	size++;
	if (size > max_allowed_size)
	  ERROR_EXIT1(-1,"trie vector grew too much (maxAllowedSize = %d)\n",
		      max_allowed_size);
	return index;
      }
      index = (index+increment) & mask;
    }
    return 0; // esto no deberia ocurrir, para que no se queje el compilador
  }

  uint32_t TrieVector::getPersistentChild (uint32_t node, uint32_t word) {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    TrieNode searched(node,word<<8);
    unsigned int index     = (node*cte_hash ^ word) & mask;
    unsigned int increment = word | 1;
    // considera que los nodos efimeros estan todos libres
    while ((data[index].wordStamp & 255) == 0 &&
	   data[index] != searched)
      index = (index+increment) & mask;
    if (data[index] != searched) {
      data[index] = searched;
      // size++;
      --max_allowed_size;
      if (size > max_allowed_size) {
	ERROR_EXIT1(-1,"trie vector grew too much (maxAllowedSize = %d)\n",
		    max_allowed_size);
      }
    }
    return index;
  }
  
  bool TrieVector::hasChild (uint32_t node, uint32_t word, uint32_t &destnode) {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    TrieNode searchedP(node,word<<8);
    TrieNode searchedE(node,(word<<8)|stamp);
    unsigned int index     = (node*cte_hash ^ word) & mask;
    unsigned int increment = word | 1;
    for (;;) {
      unsigned int stmp = data[index].wordStamp & 255;
      if (stmp == 0) { // es un nodo persistente
	if (data[index] == searchedP) { destnode=index; return true; }
      } else if (stmp == stamp) {
	if (data[index] == searchedE) { destnode=index; return true; }
      } else {
	return false;
      }
      index = (index+increment) & mask;
    }
    return false; // esto no deberia ocurrir, para que no se queje el compilador
  }

  bool TrieVector::hasSequence(const uint32_t *sequence, int length,
			       uint32_t &destnode) {
    uint32_t index = rootNode();
    for (int i=0; i<length; ++i) {
      if (!hasChild(index, sequence[i], index)) return false;
    }
    destnode = index;
    return true;
  }
  
  uint32_t TrieVector::searchSequence(const uint32_t *sequence, int length) {
    uint32_t index = rootNode();
    for (int i=0; i<length; ++i)
      index = getChild(index,sequence[i]);
    return index;
  }

  uint32_t TrieVector::searchPersistentSequence(const uint32_t *sequence, int length) {
    uint32_t index = rootNode();
    for (int i=0; i<length; ++i)
      index = getPersistentChild(index,sequence[i]);
    return index;
  }

  /// devuelve la longitud, -2 si no existe, -1 si no cabe
  int TrieVector::getSequence(uint32_t node, uint32_t *sequence, int maxLength) {
    if ((data[node].wordStamp & 255) != stamp) return -2;
    int len=0;
    while (node != 0 && len<maxLength) {
      sequence[len] = data[node].wordStamp >> 8;
      node          = data[node].parent;
      len++;
    }
    if (node != 0) return -1;
    // reverse sequence:
    int first=0,last=len-1;
    while (first<last) {
      swap(sequence[first],sequence[last]);
      first++; last--;
    }
    return len;
  }

} // namespace
