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
#ifndef FIFO_BLOCK_H
#define FIFO_BLOCK_H

#include "swap.h"

namespace april_utils {
  
  /**
   * representa una cola fifo pero es una implementación hibrida entre
   * vector y listas enlazadas. Se trata de una lista de vectores de
   * talla fija (talla fifo_block_size) y requiere dos índices de
   * vector para indicar en qué posiciones del primer vector de la
   * lista, y en qué posición del último vector de la lista, se
   * encuentran los elementos a extraer y a insertar, respectivamente.
   */
  template<typename T, int fifo_block_size=63>
  class fifo_block {
    
    struct fifo_block_node {
      T value[fifo_block_size];
      fifo_block_node *next;
      fifo_block_node() : next(0) { }
    };

    // atributos
    fifo_block_node *first, *last;
    int index_read;  // pos siguiente lectura
    int index_write; // pos siguiente escritura

    // metodo auxiliar
    void internal_copy(const fifo_block& other) {
      // asumimos que nuestro objeto está ya vacio
      int other_read = other.index_read; // ahora recorremos los elementos de other
      const fifo_block_node *other_pointer = other.first;
      while (other_pointer != other.last || other_read < other.index_write) {
	put(other_pointer->value[other_read]);
	other_read++;
	if (other_read >= fifo_block_size) {
	  other_read = 0;
	  other_pointer = other_pointer->next;
	}
      }
    }
    
  public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    struct block_iterator {
      friend class fifo_block;
    private:
      fifo_block_node *ptr;
    public:
      typedef T         value_type;
      typedef T*        pointer;
      typedef T&        reference;
      typedef const T&  const_reference;
      typedef ptrdiff_t difference_type;
      
      block_iterator(fifo_block_node *p=0): ptr(p) {}
      
      block_iterator& operator++() { // preincrement
	ptr = ptr->next; 
	return *this;
      }

      T* operator *() { // dereference
	return ptr->value;
      }
      
      bool operator == (const block_iterator &i) { return ptr == i.ptr; }
      bool operator != (const block_iterator &i) { return ptr != i.ptr; }
      
    };
    
    struct const_iterator;
    
    struct iterator {
      friend struct const_iterator;
      friend class  fifo_block;
    private:
      int index;
      fifo_block_node *ptr;
      
    public:
      typedef T         value_type;
      typedef T*        pointer;
      typedef T&        reference;
      typedef const T&  const_reference;
      typedef ptrdiff_t difference_type;
      
      iterator(): index(0),ptr(0) {}
      iterator(int i, fifo_block_node *p): index(i),ptr(p) {}
      
      iterator& operator++() { // preincrement
	index++;
	if (index >= fifo_block_size) {
	  index = 0;
	  ptr = ptr->next; 
	}
	return *this;
      }

      iterator operator++(int) { // postincrement
	int itmp              = index;
	fifo_block_node *ptmp = ptr;
	index++;
	if (index >= fifo_block_size) {
	  index = 0;
	  ptr = ptr->next; 
	}
	return iterator(itmp,ptmp);
      }

      T& operator *() { // dereference
	return ptr->value[index];
      }
      
      T* operator ->() {
	return &(ptr->value[index]);
      }
      
      bool operator == (const iterator &i) { return index == i.index && ptr == i.ptr; }
      bool operator != (const iterator &i) { return index != i.index || ptr != i.ptr; }
      
    }; // closes struct iterator
    
    struct const_iterator {
    private:
      int index;
      const fifo_block_node *ptr;
      
    public:
      typedef T         value_type;
      typedef T*        pointer;
      typedef T&        reference;
      typedef const T&  const_reference;
      typedef ptrdiff_t difference_type;
      
      const_iterator(): index(0), ptr(0) {}
      const_iterator(int i, const fifo_block_node *p): index(i), ptr(p) {}
      const_iterator(const iterator &i): index(i.index), ptr(i.ptr) {}
      
      const_iterator& operator++() { // preincrement
	index++;
	if (index >= fifo_block_size) {
	  index = 0;
	  ptr = ptr->next; 
	}
      }
      
      const_iterator operator++(int) { // postincrement
	int itmp              = index;
	fifo_block_node *ptmp = ptr;
	index++;
	if (index >= fifo_block_size) {
	  index = 0;
	  ptr = ptr->next; 
	}
	return const_iterator(itmp,ptmp);
      }

      const T& operator *() { // dereference
	return ptr->value[index];
      }

      const T* operator ->() {
	return &(ptr->value[index]);
      }

      bool operator == (const const_iterator &i) { return index == i.index && ptr == i.ptr; }
      bool operator != (const const_iterator &i) { return index != i.index || ptr != i.ptr; }

    }; // closes struct const_iterator

    iterator begin() { return iterator(index_read,first); }
    iterator end()   { return iterator(index_write,last); }
    const_iterator begin() const { return const_iterator(index_read,first); }
    const_iterator end()   const { return const_iterator(index_write,last); }

    block_iterator block_begin() { return block_iterator(first); }
    block_iterator block_end()   { return block_iterator(0); }

    size_type  size()  const { return count();  }
    size_type  block_size()  const { return fifo_block_size;  }
    size_type  max_size() const { return size_type(-1); }
    bool empty() const { return (first == last && index_read >= index_write); }
    void clear() {
      while (first) {
	fifo_block_node *aux = first;
	first = first->next;
	delete aux;
      }
      index_read = index_write = 0;
      last = 0;
    }

    bool is_end(const iterator &iter) const { 
      return (iter.ptr == 0) || ((iter.ptr == last) && (iter.index == index_write));
    }

    fifo_block(): first(0), last(0), index_read(0), index_write(0) {}    
    fifo_block(const fifo_block& other) : first(0), last(0), 
					  index_read(0), index_write(0) {
      internal_copy(other);
    }
    ~fifo_block() { clear(); }
    
    fifo_block& operator=(const fifo_block& other) { // asignment operator
      if (&other != this) { clear(); internal_copy(other); }
      return *this;
    }

    T* put(const T& value) {
      if (last == 0 || index_write >= fifo_block_size) {
	fifo_block_node *aux = new fifo_block_node;
	((last) ? last->next : first) = aux;
	last = aux;
	index_write = 0;
      }
      T *address = &(last->value[index_write]);
      last->value[index_write] = value;
      index_write++;
      return address;
    }

    void drop() { // CUIDADO! REQUIERE QUE NO ESTE VACIA
      index_read++;
      if (index_read >= fifo_block_size) {
	fifo_block_node *aux = first;
	if (last == first)
	  last = 0;
	first = first->next;
	index_read = 0;
	delete aux;
      }
    }

    bool get(T &value) {
      if (empty()) return false;
      value = first->value[index_read];
      drop();
      return true;
    }

    bool consult(T &value) {
      if (empty()) return false;
      value = first->value[index_read];
      return true;
    }

    int count() const {
      int i = index_read, c = 0;
      const fifo_block_node *p = first;
      while (p != last || i < index_write) {
	c++; i++;
	if (i >= fifo_block_size) { i = 0; p = p->next;	}
      }
      return c;      
    }

    void take_content_from(fifo_block& other) {
      if (&other != this) {
	clear();
	first       = other.first;
	last        = other.last;
	index_read  = other.index_read;
	index_write = other.index_write;
	other.first = other.last = 0;
	other.index_read = other.index_write= 0;
      }
    }

    void swap(fifo_block &other) {
      april_utils::swap(index_read,  other.index_read);
      april_utils::swap(index_write, other.index_write);
      april_utils::swap(first, other.first);
      april_utils::swap(last,  other.last);
    }

    bool contains_node(T *ptr) const {
      for (fifo_block_node *r = first; r!=0; r=r->next)
	if (r->value <= ptr && ptr < r->value+fifo_block_size)
	  return true;
      return false;
    }

  }; // closes class fifo_block

} // closes namespace april_utils

#endif // FIFO_BLOCK_H
