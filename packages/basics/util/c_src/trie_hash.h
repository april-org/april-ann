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
#ifndef TRIE_HASH_H
#define TRIE_HASH_H

#include <cstdio>
#include <cstdlib>

#include "aux_hash_table.h"
#include "error_print.h"
#include "hash_table.h"

namespace AprilUtils {

  /// Plantilla para implementar un Trie generico, que guarda
  /// transiciones entre estados con el tipo int, y da como salida el tipo
  /// O.
  template <typename O>
  class TrieHash {
  public:
    // una transicion va de (estado, int) => (dest)
    typedef AprilUtils::hash<int_pair, int> transitions_t;
  
  private:
    // tabla de finales
    AprilUtils::hash<int, O> outputs_tbl;
    
    // nodos y transiciones
    transitions_t  transitions_tbl;
    int           *levels_tbl;
    int            num_nodes;
    int            next_node;
      
  public:
    
    /////////////////////////////////////////////////////////////////////////////
    
    typedef typename AprilUtils::hash<int, O>::iterator outputs_iterator;
    
    // iterador sencillo
    class iterator {
      friend class TrieHash;
      TrieHash *trie;
      typename transitions_t::iterator hash_it;
      
      iterator(TrieHash *trie, typename transitions_t::iterator it) {
	this->trie    = trie;
	this->hash_it = it;
      }
      
    public:
      iterator(): trie(0) {}
      
      iterator& operator++(){ // preincrement
	hash_it++;
	return *this;
      }
      
      iterator operator++(int){ // postincrement
	iterator tmp(*this);
	++(*this);
	return tmp;
      }
      
      int  get_node() {
	// typedef AprilUtils::hash<(estado,word), int,
	return hash_it->first.first;
      }
      int    get_tr_value() {
	// typedef AprilUtils::hash<(estado,word), int,
	return hash_it->first.second;
      }
      int  get_dest() {
	// typedef AprilUtils::hash<(estado,word), int,
	return hash_it->second;
      }
      int get_level() {
	// typedef AprilUtils::hash<(estado,word), int,
	return trie->levels_tbl[hash_it->first.first];
      }
      pair<const int_pair, int> & get_transition() {
	return *hash_it;
      }
      const O *get_outputs() {
	// typedef AprilUtils::hash<(estado,word), int,
	return trie->outputs_tbl.find(hash_it->first.first);
      }
      
      bool operator == (const iterator &other) {
	return ((hash_it == other.hash_it) &&
		(trie    == other.trie));
      }
      
      bool operator != (const iterator &other) {
	return ((hash_it != other.hash_it) ||
		(trie    != other.trie));
      }
      
    };
    
    /////////////////////////////////////////////////////////////////////////////
    
    /// Reserva memoria, el tamanyo del Trie debe ser conocido a priori.
    TrieHash(int num_nodes = 8) :
      outputs_tbl(8, 2.0),
      transitions_tbl(8, 2.0)
    {
      if (num_nodes <= 0) num_nodes = 8;
      this->num_nodes = num_nodes;
      next_node	      = 1;
      levels_tbl      = new int[num_nodes];
      levels_tbl[0]   = 0;
    }
    
    /// Libera los recursos
    ~TrieHash() {
      delete[] levels_tbl;
    }

    void clear() {
      next_node = 1;
      transitions_tbl.clear();
      outputs_tbl.clear();
    }

    int get_state_level(int st) {
      return levels_tbl[st];
    }
    
    int get_size() { return next_node; }
    
    /// Crea una transicion entre estados
    void insert(int orig, int dest, int key, int level) {
      if (orig > next_node) next_node  = orig+1;
      if (dest > next_node) next_node  = dest+1;
      if (orig > num_nodes)
	resize(orig<<1);
      if (dest > num_nodes)
	resize(dest<<1);
      transitions_tbl[int_pair(orig, key)] = dest;
      levels_tbl[orig]                         = level;
      levels_tbl[dest]                         = level+1;
    }
    /// Guarda los outputs
    void set_state_output(int st, O value) {
      outputs_tbl[st] = value;
    }

    /// Devuelve los outputs
    const O *get_state_output(int st) {
      return outputs_tbl.find(st);
    }
    
    /// Inserta una secuencia en trie, y le asocia value como salida
    void insert(const int *key, O value, int key_size) {
      int current_node = 0;
      int next_level = 1;
      for (int i=0; i<key_size; ++i) {
	int_pair tr_key(current_node, key[i]);
	// busca la componente actual de la secuencia en el estado current_node
	const int *aux = transitions_tbl.find(tr_key);
	if (aux != 0) current_node = *aux;
	else {
	  // si no lo encuentra, crea un nuevo acceso en la tabla y un
	  // nuevo nodo destino
	  transitions_tbl[tr_key] = next_node;
	  levels_tbl[next_node]   = next_level;
	  current_node		  = next_node++;
	  if (next_node >= num_nodes)
	    resize(num_nodes<<1);
	}
	next_level++;
      }
      // guarda el valor de salida en la tabla de outputs
      outputs_tbl[current_node] = value;
    }
  
    void resize(int n) {
      // para hacer un resize del Trie...
      int    *old_levels = levels_tbl;
      int old_num_nodes  = num_nodes;
      num_nodes          = n;
      levels_tbl         = new int[n];
      for (int i=0; i<old_num_nodes; ++i)
	levels_tbl[i] = old_levels[i];
      delete[] old_levels;
    }

    bool search_transition(int current_node, int key, int &dest_node_id) {
      const int *aux = transitions_tbl.find(int_pair(current_node, key));
      if (aux == 0) return false;
      dest_node_id = *aux;
      return true;
    }
    
    /// Busca la secuencia determinada por key de tipo int *, de tamanyo
    /// key_size, y si la encuentra, y es un nodo de salida, devuelve el
    /// valor tipo O guardado en dicho nodo.
    const O *find(const int *key, int key_size) {
      int current_node = 0;
      // iteramos sobre el tamanyo de key
      for (int i=0; i<key_size; ++i) {
	const int *aux = transitions_tbl.find(int_pair(current_node, key[i]));
	if (aux == 0) return 0;
	current_node = *aux;
      }
      // buscamos el nodo en la lista de outputs del Trie
      return outputs_tbl.find(current_node);
    }

    /// Busca la secuencia determinada por key de tipo int *, de tamanyo
    /// key_size, y si la encuentra, devuelve el estado destino
    bool search_dest_state(const int *key, int key_size, int &dest_node_id) {
      int current_node = 0;
      // iteramos sobre el tamanyo de key
      for (int i=0; i<key_size; ++i) {
	const int *aux = transitions_tbl.find(int_pair(current_node, key[i]));
	if (aux == 0) return false;
	current_node = *aux;
      }
      dest_node_id = current_node;
      return true;
    }

    /// Busca la secuencia determinada por key de tipo int *, de tamanyo
    /// key_size, y si la encuentra, y es un nodo de salida, devuelve el
    /// valor tipo O guardado en dicho nodo.
    O &get(const int *key, int key_size) {
      int current_node  = 0;
      int next_level = 1;
      for (int i=0; i<key_size; ++i) {
	int_pair tr_key(current_node, key[i]);
	// busca la componente actual de la secuencia en current_node
	const int *aux = transitions_tbl.find(tr_key);
	if (aux != 0) current_node = *aux;
	else {
	  // si no lo encuentra, crea un nuevo acceso en la tabla y un
	  // nuevo nodo destino
	  transitions_tbl[tr_key] = next_node;
	  levels_tbl[next_node]   = next_level;
	  current_node		  = next_node++;
	  if (next_node >= num_nodes)
	    resize(num_nodes<<1);
	}
	next_level++;
      }
      // buscamos el nodo en la lista de outputs del Trie
      return outputs_tbl[current_node];
    }
    
    iterator begin() {
      return iterator(this, transitions_tbl.begin());
    }
    iterator end() {
      return iterator(this, transitions_tbl.end());
    }

    outputs_iterator begin_outputs() {
      return outputs_tbl.begin();
    }
    outputs_iterator end_outputs() {
      return outputs_tbl.end();
    }
  };

};

#endif // TRIE_HASH_H
