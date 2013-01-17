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
#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include "aux_hash_table.h"
#include <cmath> // ceilf
#include <stddef.h> // ptrdiff_t, size_t...
#include "pair.h"
//#include <iterator>
#include "vector.h"

/** \namespace april_utils 

    esto es una prueba

*/

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
	    typename HashFcn  = default_hash_function<KeyType>,
	    typename EqualKey = default_equality_comparison_function<KeyType> >
  class hash {
  public:
    
    // Para ser un "Pair associative container" el value_type deberia ser
    // un std::pair.
    typedef KeyType         key_type;
    typedef DataType        data_type;
    typedef pair<const key_type, data_type> 
    value_type;
    typedef value_type&     reference;
    typedef const reference const_reference;
    typedef value_type*     pointer;
    typedef ptrdiff_t       difference_type;
    typedef size_t          size_type;

    struct node {
      value_type value;
      node *next;
      node(const KeyType& k, const DataType& d, node *n):
	value(k,d), next(n) {}
      node(const node& other):
	value(other.value.first, other.value.second), next(other.next){}
    };

  private:
    node **buckets; // vector of pointers to node
    int num_buckets;// a power of 2
    int hash_mask;  // useful to do "& hash_mask" instead of "% num_buckets"
    int the_size;
    int rehash_threshold;
    float max_load_factor; // for rehashing purposes
    int first_power_of_2(int value);// auxiliary function
    void copy_buckets(const hash&); // auxiliary function
    HashFcn hash_function;
    EqualKey equal_key;
  public:
    hash(int nbckts=8, float mxloadf=4.0); // constructor
    hash(const hash&);   // copy constructor
    ~hash(); // destructor
    hash& operator=(const hash&);  // assignment operator
      
    void clear(); // Erases all of the elements.
    data_type* find(const key_type& k);
    value_type* find_pair(const key_type& k) const;
    bool search(const key_type& k) const {
      return find_pair(k) != 0;
    }
    void erase(const key_type& k);

    template<typename predicate> void delete_if(predicate &p);

    // bool operator==(const hash&) const; // tests two hash for equality.
    value_type* find_and_add_pair(const key_type& k, bool &isNew);
    data_type& operator[] (const key_type& k) {
      bool dummy;
      return find_and_add_pair(k,dummy)->second;
    }
    // devuelve true si se acaba de crear
    bool insert(const key_type& k, const data_type& d) {
      bool resul;
      find_and_add_pair(k,resul)->second = d;
      return resul;
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
    float get_max_load_factor() const {
      return max_load_factor;
    }
    int get_num_buckets() const {
      return num_buckets;
    }
    void get_hash_nodes(vector<pointer> &output_vector) const;

    class const_iterator;

    class iterator {
      friend class const_iterator;
      friend class hash;

      hash *h;
      int index;
      node *ptr;

      iterator(hash *h, int index, node *ptr):
	h(h), index(index), ptr(ptr) {}

    public:
      // Para ser un "Pair associative container" el value_type deberia ser
      // un std::pair. Bueno, que le vamos a hacer :_(
      typedef KeyType         key_type;
      typedef DataType        data_type;
      typedef pair<const key_type, data_type> 
      value_type;
      typedef value_type&     reference;
      typedef const reference const_reference;
      typedef value_type*     pointer;
      typedef ptrdiff_t       difference_type;
      typedef size_t          size_type;
      //typedef std::forward_iterator_tag iterator_category;
	
      iterator(): h(0), index(0), ptr(0) {}
      iterator(const iterator &other):
	h(other.h), index(other.index), ptr(other.ptr) {}
      iterator& operator=(const iterator &other) {
	if (&other != this) {
	  h     = other.h;
	  index = other.index;
	  ptr   = other.ptr;
	}

	return *this;
      }

      iterator& operator++(){ // preincrement
	ptr = ptr->next;
	while (ptr == 0) {
	  index++;
	  if (index < h->num_buckets)
	    ptr = h->buckets[index];
	  else break;
	}
	return *this;
      }

      iterator operator++(int){ // postincrement
	iterator tmp(*this);
	++(*this);
	return tmp;
      }

      value_type& operator *() { //dereference
	return ptr->value;
      }

      value_type* operator ->() {
	return &(ptr->value);
      }

      bool operator == (const iterator &other) {
	return ((ptr == other.ptr) && (h == other.h) 
		&& (index == other.index));
      }

      bool operator != (const iterator &other) {
	return ((ptr != other.ptr) || (h != other.h)
		|| (index != other.index));
      }

    };
      
    class const_iterator {
      friend class hash;

      const hash *h;
      int index;
      const node *ptr;
        
      const_iterator(hash *h, int index, node *ptr):
	h(h), index(index), ptr(ptr) {}

    public:
      // Para ser un "Pair associative container" el value_type deberia ser
      // un std::pair. Bueno, que le vamos a hacer :_(
      typedef KeyType         key_type;
      typedef DataType        data_type;
      typedef pair<const key_type, data_type> 
      value_type;
      typedef value_type&     reference;
      typedef const reference const_reference;
      typedef value_type*     pointer;
      typedef ptrdiff_t       difference_type;
      typedef size_t          size_type;
      //typedef std::forward_iterator_tag iterator_category;

      const_iterator(): h(0), index(0), ptr(0) {}
        
      // Copy constructor (from iterator and const_iterator)
      const_iterator(const iterator &other):
	h(other.h), index(other.index), ptr(other.ptr) {}
      const_iterator(const const_iterator &other):
	h(other.h), index(other.index), ptr(other.ptr) {}

      // Assignment operators
      const_iterator& operator=(const iterator &other) {
	h     = other.h;
	index = other.index;
	ptr   = other.ptr;
	return *this;
      }
      const_iterator& operator=(const const_iterator &other) {
	h     = other.h;
	index = other.index;
	ptr   = other.ptr;
	return *this;
      }

      const_iterator& operator++(){ // preincrement
	ptr = ptr->next;
	while (ptr == 0) {
	  index++;
	  if (index < h->num_buckets)
	    ptr = h->buckets[index];
	  else break;
	}
	return *this;
      }

      const_iterator operator++(int){ // postincrement
	const_iterator tmp(*this);
	++(*this);
	return tmp;
      }

      const value_type& operator *() { //dereference
	return ptr->value;
      }

      const value_type* operator ->() {
	return &(ptr->value);
      }

      // Eq comparisons
      bool operator == (const iterator &other) const {
	return ((ptr == other.ptr) && (h == other.h) 
		&& (index == other.index));
      }
      bool operator == (const const_iterator &other) const {
	return ((ptr == other.ptr) && (h == other.h) 
		&& (index == other.index));
      }
      bool operator != (const iterator &other) const {
	return ((ptr != other.ptr) || (h != other.h)
		|| (index != other.index));
      }
      bool operator != (const const_iterator &other) const {
	return ((ptr != other.ptr) || (h != other.h)
		|| (index != other.index));
      }

    };

    iterator begin() {
      int i=0;
      node *ptr = buckets[0];
      while (ptr == 0) {
	i++;
	if (i < num_buckets)
	  ptr = buckets[i];
	else break;
      }

      return iterator(this, i, ptr);
    }

  private:
    iterator end_iterator;
    
  public:
    iterator end() { return end_iterator; }
  };

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  int hash<ky,dt,hfcn,eqky>::first_power_of_2(int value) {
    int result = 1;
    while (result < value)
      result += result;
    return result;
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  hash<ky,dt,hfcn,eqky>::hash(int nbckts, float mxloadf) {
    num_buckets     = first_power_of_2(nbckts);
    hash_mask       = num_buckets-1; // num_buckets is a power of 2
    the_size        = 0;
    max_load_factor = mxloadf;
    rehash_threshold= (int)ceilf(max_load_factor*num_buckets);
    buckets         = new node*[num_buckets];
    end_iterator    = iterator(this, num_buckets, 0);
    for (int i=0; i<num_buckets; i++)
      buckets[i] = 0;
    
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  inline void hash<ky,dt,hfcn,eqky>::copy_buckets(const hash& other) { // auxiliary function
    // num_buckets == other.num_buckets is assumed
    for (int i=0; i<num_buckets; i++) {
      node **totail = &buckets[i];
      for (node *reco = other.buckets[i]; 
	   reco != 0;
	   reco = reco->next) {
	*totail = new node(*reco);
	// **totail = *reco; // shallow copy
	totail = &((*totail)->next);
      }
      *totail = 0;
    }
  }

  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  hash<ky,dt,hfcn,eqky>::hash(const hash& other) { // copy constructor
    num_buckets     = other.num_buckets;
    hash_mask       = other.hash_mask;
    the_size        = other.the_size;
    max_load_factor = other.max_load_factor;
    rehash_threshold= other.rehash_threshold;
    buckets         = new node* [num_buckets];
    end_iterator    = iterator(this, num_buckets, 0);
    copy_buckets(other);
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void hash<ky,dt,hfcn,eqky>::clear() { // Erases all of the elements.
    if (!empty()) {
      for (int i=0; i<num_buckets; i++)
	while (buckets[i]) {
	  node *aux =buckets[i];
	  buckets[i] = buckets[i]->next;
	  delete aux;
	}
      the_size = 0;
    }
  }

  // recibe un vector y deposita en el las direcciones de todos los
  // nodos de la tabla, alternativa a usar un iterador para recorrer
  // toda la tabla, que quizas seria lo mas elegante y adecuado
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void hash<ky,dt,hfcn,eqky>::get_hash_nodes(vector<pointer> &output_vector) const {
    output_vector.clear();
    if (!empty()) {
      for (int i=0; i<num_buckets; i++)
	for (node * r = buckets[i]; r!=0; r=r->next)
	  output_vector.push_back(&r->value);
    }
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  hash<ky,dt,hfcn,eqky>& 
  hash<ky,dt,hfcn,eqky>::operator=(const hash& other) {  // The assignment operator
    if (this != &other) {
      clear();
      if (num_buckets != other.num_buckets) {
	delete[] buckets;
	num_buckets  = other.num_buckets;
	buckets      = new node* [num_buckets];
      }
      end_iterator    = iterator(this, num_buckets, 0);
      hash_mask       = other.hash_mask;
      the_size        = other.the_size;
      max_load_factor = other.max_load_factor;
      rehash_threshold= other.rehash_threshold;
      copy_buckets(other);
    }
    return *this;
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  hash<ky,dt,hfcn,eqky>::~hash() { // destructor
    clear();
    delete[] buckets;
  }
  
  // template <typename ky, typename dt, 
  // 	  typename hfcn, typename eqky>
  // bool hash<ky,dt,hfcn,eqky>::operator==(const hash&) const { // Tests two hash for equality.
  //   if (size() != other.size()) return false;
  //   // equal size, look for inclusion
  //   for (int i=0; i<num_buckets; i++)
  //     for (const node *reco = buckets[i]; 
  // 	 reco != 0;
  // 	 reco = reco->next) {
  //       const node *p = other.find(reco->value.first);
  //       if (!p || reco->value.first != p->value.first) // falta comparar campo data!!!!
  // 	return false;
  //     }
  //   return true;
  // }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  dt* hash<ky,dt,hfcn,eqky>::find(const ky& k) {
    value_type* p = find_pair(k);
    return p ? &(p->second) : 0;
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  pair<const ky,dt>* hash<ky,dt,hfcn,eqky>::find_pair(const ky& k) const {
    unsigned int index = hash_function(k) & hash_mask;
    for (node* reco = buckets[index]; 
	 reco != 0;
	 reco = reco->next)
      if (equal_key(k,reco->value.first)) 
	return &(reco->value);
    return 0;
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void hash<ky,dt,hfcn,eqky>::erase(const ky& k) {
    unsigned int index = hash_function(k) & hash_mask;
    for (node** reco = &buckets[index]; 
	 *reco != 0;
	 reco = &((*reco)->next))
      if (equal_key(k,(*reco)->value.first)) {
	node* aux = *reco;
	*reco = (*reco)->next;
	delete aux;
	the_size--;
	return;
      }
    return;
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  pair<const ky,dt>* hash<ky,dt,hfcn,eqky>::find_and_add_pair(const ky& k, bool &isNew) {
    unsigned int index = hash_function(k) & hash_mask;
    for (node* reco = buckets[index]; 
	 reco != 0;
	 reco = reco->next)
      if (equal_key(k,reco->value.first)) {
	isNew = false;
	return &(reco->value);
      }
    node* newnode = new node(k, dt(), buckets[index]);
    buckets[index]= newnode;
    the_size++;
    if (the_size >= rehash_threshold) // rehashing?
      resize(2*num_buckets);
    isNew = true;
    return &(newnode->value);
  }
  
  template <typename ky, typename dt, 
	    typename hfcn, typename eqky>
  void hash<ky,dt,hfcn,eqky>::resize(int n) {
    n = first_power_of_2(n);
    if (n != num_buckets) {
      node **old_buckets = buckets;
      int old_num_buckets= num_buckets;
      num_buckets = n;
      end_iterator= iterator(this, num_buckets, 0);
      hash_mask   = n-1;
      rehash_threshold= (int)ceilf(max_load_factor*num_buckets);
      buckets     = new node* [num_buckets];
      for (int i=0; i<num_buckets; i++)
	buckets[i] = 0;
      for (int i=0; i<old_num_buckets; i++)
	while (old_buckets[i]) {
	  node *aux = old_buckets[i];
	  old_buckets[i] = old_buckets[i]->next;
	  unsigned int index = hash_function(aux->value.first) & hash_mask;
	  aux->next = buckets[index];
	  buckets[index] = aux;
	}
      delete[] old_buckets;
    }
  }
  
  template <typename ky, typename dt,typename hfcn, typename eqky>
  template <typename predicate>
  void hash<ky,dt,hfcn,eqky>::delete_if(predicate &p) {
    for (int i=0; i<num_buckets; i++) {
      node *cur = buckets[i];
      node *prev = 0;
      while (cur != 0) {
	if (p(cur->value)) {
	  if (prev == 0)
	    buckets[i] = cur->next;
	  else
	    prev->next = cur->next;
	  
	  node *tmp = cur;
	  cur = cur->next; // prev sigue siendo el mismo al borrar cur
	  delete tmp;
	} else {
	  prev = cur;
	  cur  = cur->next;
	}
      }
    }
    
  }
  
} // namespace april_utils

#endif // HASH_TABLE_H
