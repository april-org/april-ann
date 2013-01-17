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
#ifndef CACHE_OPEN_ADDR_HASH_H
#define CACHE_OPEN_ADDR_HASH_H

/* Cache con tabla hash de direccionamiento abierto (CON TIMESTAMP)
   
   Se utiliza un vector para guardar directamente los valores. Cuando
   se busca un valor, se utiliza la función de dispersión para obtener
   el indice en el vector.

   Cuando  una posicion  es ocupada,  la clave  y el  valor  viejos se
   sobreescriben con  los nuevos.
   
   Los tipos de datos para los campos clave y valor asumen que el
   operador de copia funciona suficientemente bien como para que al
   "machacar" una posición del vector con un valor nuevo, el valor
   anterior se libere convenientemente.

   También se asume que ambos tipos de datos tienen constructores por
   defecto.

   Un valor estará siempre en la misma posición ya que no hay
   rehashing.

*/

#include "aux_hash_table.h"
#include <cmath> // ceilf
#include <stddef.h> // ptrdiff_t, size_t...
#include "pair.h"

namespace april_utils {
  
  // HashFcn es del tipo:
  // struct hash_fcn {
  //   unsigned int operator()(const char* s1) const {
  //     return blah...;
  //   }
  // };
  
  // EqualKey es del tipo:
  // struct eqstr {
  //   bool operator()(const char* s1, const char* s2) const {
  //     return strcmp(s1, s2) == 0;
  //   }
  // };

  template <typename KeyType, typename DataType, 
	    typename HashFcn = default_hash_function<KeyType>, 
            typename EqualKey = default_equality_comparison_function<KeyType> >
  class cache_open_addr_hash {
  public:
    
    struct node {
      pair<KeyType,DataType> value;
      unsigned int           stamp;
      node() {}
      node(KeyType& k, DataType& d):
	value(k,d) {}
    };

    typedef KeyType         key_type;
    typedef DataType        data_type;
    typedef pair<const key_type, data_type> value_type;
    typedef value_type&     reference;
    typedef const reference const_reference;
    typedef value_type*     pointer;
    typedef ptrdiff_t       difference_type;
    typedef size_t          size_type;

  private:
    static const unsigned int maxstamp = 1<<30;
    unsigned int timestamp;
    node *buckets;  // vector of nodes
    int used_size;
    int num_buckets;// a power of 2
    int hash_mask;  // useful to do "& hash_mask" instead of "%
		    // num_buckets"
    int first_power_of_2(int value);// auxiliary function
    HashFcn hash_function;
    EqualKey equal_key;
  public:
    cache_open_addr_hash(int nbckts=18); // constructor
    cache_open_addr_hash(const cache_open_addr_hash&); // copy constructor
    ~cache_open_addr_hash(); // destructor
    // assignment operator
    cache_open_addr_hash& operator=(const cache_open_addr_hash&);
      
    void clear(); // Erases all of the elements.
    data_type* find(const key_type& k);
    const value_type* find_pair(const key_type& k) const;
    bool search(const key_type& k) const {
      return find_pair(k) != 0;
    }
    data_type& operator[] (const key_type& k);
    void insert(const key_type& k, data_type& d) {
      (*this)[k] = d;
    }

    int bucket_count() const {
      // Returns the number of buckets used by the hash
      return used_size;
    }
    
    bool get(const key_type&	k,
	     data_type**        d);

    class const_iterator;
    
    class iterator {
      friend class const_iterator;
      friend class cache_open_addr_hash;

      cache_open_addr_hash *h;
      int index;

      iterator(cache_open_addr_hash *h, int index):
	h(h), index(index) {}

    public:
      typedef KeyType         key_type;
      typedef DataType        data_type;
      typedef pair<const key_type, data_type> value_type;
      typedef value_type&     reference;
      typedef const reference const_reference;
      typedef value_type*     pointer;
      typedef ptrdiff_t       difference_type;
      typedef size_t          size_type;

      iterator(): h(0), index(0) {}
      iterator(const iterator &other):
	h(other.h), index(other.index) {}
      iterator& operator=(const iterator &other) {
	if (&other != this) {
	  h     = other.h;
	  index = other.index;
	}
	return (*this);
      }

      iterator& operator++() { // preincrement
	// asumimos que no se incrementa un iterador con
	// index>=num_buckets
	do {
	  index++;
	} while (index<h->num_buckets &&
		 h->buckets[index].stamp != h->timestamp);
	return *this;
      }

      iterator operator++(int) { // postincrement
	iterator tmp(*this);
	++(*this);
	return tmp;
      }

      value_type& operator *() { //dereference
	return h->buckets[index].value;
      }

      value_type* operator ->() {
	return &(h->buckets[index].value);
      }

      bool operator == (const iterator &other) {
	return ((index == other.index) && (h == other.h));
      }

      bool operator != (const iterator &other) {
	return ((index != other.index) || (h != other.h));
      }

    };
      
    class const_iterator {
      friend class cache_open_addr_hash;

      const cache_open_addr_hash *h;
      int index;
        
      const_iterator(cache_open_addr_hash *h, int index):
	h(h), index(index) {}

    public:
      typedef KeyType         key_type;
      typedef DataType        data_type;
      typedef pair<key_type, data_type> value_type;
      typedef value_type&     reference;
      typedef const reference const_reference;
      typedef value_type*     pointer;
      typedef ptrdiff_t       difference_type;
      typedef size_t          size_type;

      const_iterator(): h(0), index(0) {}
        
      // Copy constructor (from iterator and const_iterator)
      const_iterator(const iterator &other):
	h(other.h), index(other.index) {}
      const_iterator(const const_iterator &other):
	h(other.h), index(other.index) {}

      // Assignment operators
      const_iterator& operator=(const iterator &other) {
	h     = other.h;
	index = other.index;
      }
      const_iterator& operator=(const const_iterator &other) {
	h     = other.h;
	index = other.index;
      }

      const_iterator& operator++() { // preincrement
	// asumimos que no se incrementa un iterador con
	// index>=num_buckets
	do {
	  index++;
	} while (index<h->num_buckets &&
		 h->buckets[index].stamp != h->timestamp);
	return *this;
      }

      const_iterator operator++(int) { // postincrement
	const_iterator tmp(*this);
	++(*this);
	return tmp;
      }

      const value_type& operator *() { //dereference
	return h->buckets[index].value;
      }

      const value_type* operator ->() {
	return &(h->buckets[index].value);
      }

      // Eq comparisons
      bool operator == (const iterator &other) {
	return ((index == other.index) && (h == other.h));
      }
      bool operator == (const const_iterator &other) {
	return ((index == other.index) && (h == other.h));
      }
      bool operator != (const iterator &other) {
	return ((index != other.index) || (h != other.h));
      }
      bool operator != (const const_iterator &other) {
	return ((index != other.index) || (h != other.h));
      }

    };

    iterator begin() {
      int i=0;
      while (i<num_buckets &&
	     buckets[i].stamp != timestamp) ++i;
      return iterator(this, i);
    }

  private:
    iterator end_iterator;
    
  public:
    iterator& end() { return end_iterator; }
      
  };
 

  template <typename ky, typename dt,
	    typename hfcn, typename eqky>
  int cache_open_addr_hash<ky,dt,hfcn,eqky>::first_power_of_2(int value) {
    int result = 4; // ojito, minimo 4. TODO: se queda este valor?
    while (result < value)
      result += result;
    return result;
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  cache_open_addr_hash<ky,dt,hfcn,eqky>::cache_open_addr_hash(int nbckts) {
    num_buckets     = first_power_of_2(nbckts);
    hash_mask       = num_buckets-1; // num_buckets is a power of 2
    end_iterator    = iterator(this, num_buckets);
    buckets         = new node[num_buckets];
    timestamp       = maxstamp - 1;
    // se encarga de inicializar el stamp
    clear();
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  cache_open_addr_hash<ky,dt,hfcn,eqky>::cache_open_addr_hash(const cache_open_addr_hash& other) { // copy constructor
    used_size       = other.used_size;
    num_buckets     = other.num_buckets;
    hash_mask       = other.hash_mask;
    buckets         = new node[num_buckets];
    for (int i=0; i<num_buckets; i++)
      buckets[i] = other.buckets[i];    
    end_iterator    = iterator(this, num_buckets);
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void cache_open_addr_hash<ky,dt,hfcn,eqky>::clear() {
    used_size = 0;
    ++timestamp;
    if (timestamp == maxstamp) {
      for (int i=0; i<num_buckets; ++i)
	buckets[i].stamp = 0;
      timestamp = 1;
    }
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  cache_open_addr_hash<ky,dt,hfcn,eqky>& 
  cache_open_addr_hash<ky,dt,hfcn,eqky>::operator=(const cache_open_addr_hash& other) {  // The assignment operator
    if (this != &other) {
      used_size = other.used_size;
      if (num_buckets != other.num_buckets) {
	delete[] buckets;
	num_buckets  = other.num_buckets;
	buckets      = new node[num_buckets];
      }
      hash_mask       = other.hash_mask;
      for (int i=0; i<num_buckets; i++)
	buckets[i] = other.buckets[i];
      end_iterator    = iterator(this, num_buckets);
    }
    return *this;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  cache_open_addr_hash<ky,dt,hfcn,eqky>::~cache_open_addr_hash() { // destructor
    delete[] buckets;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
   dt* cache_open_addr_hash<ky,dt,hfcn,eqky>::find(const ky& k) {
    value_type* p = const_cast<value_type*>(find_pair(k));
    return p ? &(p->second) : 0;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
    const pair<const ky,dt> * cache_open_addr_hash<ky,dt,hfcn,eqky>::find_pair(const ky& k) const {
    unsigned int index = hash_function(k) & hash_mask;
    if (equal_key(buckets[index].value.first,k) &&
	buckets[index].stamp == timestamp)
      return &(buckets[index].value);
    return 0;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  dt& cache_open_addr_hash<ky,dt,hfcn,eqky>::operator[] (const ky& k) {
    unsigned int index = hash_function(k) & hash_mask;
    if (equal_key(buckets[index].value.first,k) &&
	buckets[index].stamp == timestamp)
      return buckets[index].value.second;
    else {
      if (buckets[index].stamp != timestamp) ++used_size;
      buckets[index].value.second = dt();
    }
    buckets[index].stamp       = timestamp;
    buckets[index].value.first = k;
    return buckets[index].value.second;
  }

  template <typename ky,   typename dt, 
	    typename hfcn, typename eqky>
  bool cache_open_addr_hash<ky,dt,hfcn,eqky>::get(const ky& k,
						  dt** d) {
    unsigned int index = hash_function(k) & hash_mask;
    // devolvemos el puntero al valor que hay en index
    *d = &buckets[index].value.second;
    if (equal_key(buckets[index].value.first,k) &&
	buckets[index].stamp == timestamp)
      return true;
    // copiamos la nueva key
    buckets[index].value.first = k;
    buckets[index].stamp       = timestamp;
    ++used_size;
    return false;
  }

} // namespace april_utils

#endif // CACHE_OPEN_ADDR_HASH_H
