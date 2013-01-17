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
#ifndef OPEN_ADDR_HASH_H
#define OPEN_ADDR_HASH_H

/* Tabla hash con direccionamiento abierto SIN timestamps ni otras
   flotituras.

   Se utiliza un vector para guardar directamente los valores. Cuando
   se busca un valor, se utiliza la función de dispersión para obtener
   2 valores: el indice en el vector y un offset para ir avanzando
   (modulo talla del vector) mientras las posiciones consultadas esten
   ocupadas por valores distintos del elemento buscado. La busqueda
   finaliza cuando se encuentra una posición libre o bien el elemento
   buscado.

   Es una plantilla y se especializa pasandole el tipo de clave y
   valor, asi como mecanismos para dispersar una clave, compararla con
   otra y ver si esta vacia.

   Los tipos de datos para los campos clave y valor asumen que el
   operador de copia funciona suficientemente bien como para que al
   "machacar" una posición del vector con un valor nuevo, el valor
   anterior se libere convenientemente.

   Tambien se asume que ambos tipos de datos tienen constructores por
   defecto.

   No se puede asumir que un valor estara siempre en la misma
   posicion, al menos tras realizar inserciones, puesto que vamos a
   implementar rehashing.

   No existe la operacion "erase" para borrar individualmente una
   entrada.

*/


#include <cmath> // ceilf
#include <stddef.h> // ptrdiff_t, size_t...
#include "aux_hash_table.h"
#include "pair.h"

namespace april_utils {
  
  // HashFcn es del tipo:
  // struct hash_fcn {
  //   unsigned int operator()(const KeyType &value) const {
  //     ...
  //   }
  // };
  
  // EqualKey es del tipo:
  // struct eqstr {
  //   bool operator()(const KeyType &valueA, const KeyType &valueB) const {
  //     ...
  //   }
  // };

  template <typename KeyType,
            typename DataType, 
	    typename HashFcn = default_hash_function<KeyType>,
            typename EqualKey = default_equality_comparison_function <KeyType> >
  class open_addr_hash {
  public:

    typedef pair<const KeyType,DataType> node;

    typedef KeyType         key_type;
    typedef DataType        data_type;
    typedef pair<const key_type, data_type> value_type;
    typedef value_type&     reference;
    typedef const reference const_reference;
    typedef value_type*     pointer;
    typedef ptrdiff_t       difference_type;
    typedef size_t          size_type;

  private:
    int      first_power_of_2(int value); // auxiliary function
    node    *buckets;    // vector of nodes
    int      num_buckets;// a power of 2
    int      hash_mask;  // useful to do "& hash_mask" instead of "% num_buckets"
    int      the_size;   // number of inserted elements
    int      rehash_threshold;
    float    max_load_factor; // for rehashing purposes, MUST BE <1.0
    HashFcn  hash_function;
    EqualKey equal_key;
    KeyType  empty_key;
  public:
    open_addr_hash(KeyType empty_key, int nbckts=8, float mxloadf=0.75); // constructor
    open_addr_hash(const open_addr_hash&);   // copy constructor
    ~open_addr_hash(); // destructor
    open_addr_hash& operator=(const open_addr_hash&);  // assignment operator
      
    void clear(); // Erases all of the elements.
    data_type* find(const key_type& k);
    const value_type* find_pair(const key_type& k) const;
    bool search(const key_type& k) const {
      return const_cast<const data_type*>(find_pair(k)) != 0;
    }
    data_type& operator[] (const key_type& k);
    void insert(const key_type& k, const data_type& d) {
      (*this)[k] = d;
    }

    void resize(int n); // rehash, num_buckets at least n after resize
    int bucket_count() const { // Returns the number of buckets used by the hash
      return num_buckets;
    }
    int size() const { // Returns the size of the hash
      return the_size;
    }
    bool empty() const {
      return the_size == 0;
    }

    class const_iterator;
    
    class iterator {
      friend class const_iterator;
      friend class open_addr_hash;

      open_addr_hash *h;
      int index;

      iterator(open_addr_hash *h, int index):
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
	return *this;
      }

      iterator& operator++() { // preincrement
	// asumimos que no se incrementa un iterador con
	// index>=num_buckets
	do {
	  index++;
	} while (index<h->num_buckets &&
		 h->equal_key(h->buckets[index].first,h->empty_key));
	return *this;
      }

      iterator operator++(int) { // postincrement
	iterator tmp(*this);
	++(*this);
	return tmp;
      }

      value_type& operator *() { //dereference
	return h->buckets[index];
      }

      value_type* operator ->() {
	return &(h->buckets[index]);
      }

      bool operator == (const iterator &other) {
	return ((index == other.index) && (h == other.h));
      }

      bool operator != (const iterator &other) {
	return ((index != other.index) || (h != other.h));
      }

    };
      
    class const_iterator {
      friend class open_addr_hash;

      const open_addr_hash *h;
      int index;
        
      const_iterator(open_addr_hash *h, int index):
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
		 h->equal_key(h->buckets[index].first,h->empty_key));
	return *this;
      }

      const_iterator operator++(int) { // postincrement
	const_iterator tmp(*this);
	++(*this);
	return tmp;
      }

      const value_type& operator *() { //dereference
	return h->buckets[index];
      }

      const value_type* operator ->() {
	return &(h->buckets[index]);
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
	     equal_key(buckets[i].first,empty_key)) i++;
      return iterator(this, i);
    }

  private:
    iterator end_iterator;
    
  public:
    iterator& end() { return end_iterator; }
      
  };
 

  template <typename ky, typename dt,
	    typename hfcn, typename eqky>
  int open_addr_hash<ky,dt,hfcn,eqky>::first_power_of_2(int value) {
    int result = 4; // ojito, minimo 4. TODO: se queda este valor?
    while (result < value)
      result += result;
    return result;
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  open_addr_hash<ky,dt,hfcn,eqky>::open_addr_hash(ky empty_key, 
						  int nbckts, float mxloadf) {
    this->empty_key = empty_key;
    num_buckets     = first_power_of_2(nbckts);
    hash_mask       = num_buckets-1; // num_buckets is a power of 2
    the_size        = 0;
    max_load_factor = (mxloadf <= 0.9) ? mxloadf : 0.9; // must be <1
    rehash_threshold= (int)ceilf(max_load_factor*num_buckets);
    end_iterator    = iterator(this, num_buckets);
    buckets         = reinterpret_cast<node*>(new pair<ky,dt>[num_buckets]);
    for (int i=0; i<num_buckets; i++)
      *const_cast<ky*>(&buckets[i].first) = empty_key;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  open_addr_hash<ky,dt,hfcn,eqky>::open_addr_hash(const open_addr_hash& other) { // copy constructor
    num_buckets     = other.num_buckets;
    hash_mask       = other.hash_mask;
    the_size        = other.the_size;
    max_load_factor = other.max_load_factor;
    rehash_threshold= other.rehash_threshold;
    buckets         = new node[num_buckets];
    for (int i=0; i<num_buckets; ++i)
      buckets[i] = other.buckets[i];    
    end_iterator    = iterator(this, num_buckets);
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void open_addr_hash<ky,dt,hfcn,eqky>::clear() {
    for (int i=0; i<num_buckets; ++i)
      *const_cast<ky*>(&buckets[i].first) = empty_key;
    the_size = 0;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  open_addr_hash<ky,dt,hfcn,eqky>& 
  open_addr_hash<ky,dt,hfcn,eqky>::operator=(const open_addr_hash& other) {  // The assignment operator
    if (this != &other) {
      if (num_buckets != other.num_buckets) {
	delete[] buckets;
	num_buckets  = other.num_buckets;
	buckets      = new node[num_buckets];
      }
      hash_mask       = other.hash_mask;
      the_size        = other.the_size;
      max_load_factor = other.max_load_factor;
      rehash_threshold= other.rehash_threshold;
      for (int i=0; i<num_buckets; ++i)
	buckets[i] = other.buckets[i];
      end_iterator    = iterator(this, num_buckets);
    }
    return *this;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  open_addr_hash<ky,dt,hfcn,eqky>::~open_addr_hash() { // destructor
    delete[] buckets;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
   dt* open_addr_hash<ky,dt,hfcn,eqky>::find(const ky& k) {
    value_type* p = const_cast<value_type*>(find_pair(k));
    return p ? &(p->second) : 0;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  const pair<const ky,dt>* open_addr_hash<ky,dt,hfcn,eqky>::find_pair(const ky& k) const {
    unsigned int hfval = hash_function(k);
    unsigned int index = hfval & hash_mask;
    unsigned int step  = (hfval >> 15) | 1; // debe ser impar
    while (!equal_key(buckets[index].first,empty_key)) {
      if (equal_key(buckets[index].first,k))
	return &(buckets[index]);
      index = (index + step) & hash_mask;
    }
    return 0;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  dt& open_addr_hash<ky,dt,hfcn,eqky>::operator[] (const ky& k) {
    unsigned int hfval = hash_function(k);
    unsigned int index = hfval & hash_mask;
    unsigned int step  = (hfval >> 15) | 1; // debe ser impar
    while (!equal_key(buckets[index].first,empty_key)) {
      if (equal_key(buckets[index].first,k))
	return buckets[index].second;
      index = (index + step) & hash_mask;
    }
    the_size++;
    if (the_size >= rehash_threshold) {
      resize(2*num_buckets);
      // recalcular index sabiendo que no esta, hfval ya lo tenemos
      index = hfval & hash_mask;
      step  = (hfval >> 15) | 1;
      while (!equal_key(buckets[index].first,empty_key))
	index = (index + step) & hash_mask;
    }
    *const_cast<ky*>(&buckets[index].first) = k;
    return buckets[index].second;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void open_addr_hash<ky,dt,hfcn,eqky>::resize(int n) {
    n = first_power_of_2(n);
    if (n != num_buckets) {
      node *old_buckets   = buckets;
      int old_num_buckets = num_buckets;
      num_buckets = n;
      end_iterator= iterator(this, num_buckets);
      hash_mask   = n-1;
      rehash_threshold= (int)ceilf(max_load_factor*num_buckets);
      buckets     = reinterpret_cast<node*>(new pair<ky,dt>[num_buckets]);
      for (int i=0; i<num_buckets; i++)
	*const_cast<ky*>(&buckets[i].first) = empty_key;
      for (int i=0; i<old_num_buckets; ++i)
	if (!equal_key(old_buckets[i].first,empty_key)) {
	  // insertamos sabiendo que no estaba
	  unsigned int hfval = hash_function(old_buckets[i].first);
	  unsigned int index = hfval & hash_mask;
	  unsigned int step  = (hfval >> 15) | 1; // debe ser impar
	  while (!equal_key(buckets[index].first,empty_key)) {
	    index = (index + step) & hash_mask;
	  }
	  *const_cast<ky*>(&buckets[index].first) = old_buckets[i].first;
          buckets[index].second = old_buckets[i].second;
	}
      delete[] old_buckets;
    }
  }

} // namespace april_utils

#endif // OPEN_ADDR_HASH_H

