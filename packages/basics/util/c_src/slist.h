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
#ifndef SLIST_H
#define SLIST_H

#include <stddef.h>
#include "swap.h"
//#include <iterator>

namespace AprilUtils {

  // Be careful! It's not std::slist :)
  // Implements Container and Front Insertion Sequence  
  template<class T> class slist {
    // Linked-list nodes
    struct node {
      T     data;
      node *next;
      // Constructor 
      node(T t, node* n):data(t), next(n) {}
    };


    node *first;
    node *last;
    size_t list_size;

    void init_list() { 
      first = 0;
      last = 0;
      list_size = 0; 
    }

    void delete_list() {  
      while (!empty())
        pop_front();

      list_size = 0;
      first = 0;
      last = 0;
    }  

    public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    struct const_iterator;

    struct iterator {
      friend struct const_iterator;
      private:
        node *ptr;

      public:
        typedef T         value_type;
        typedef T*        pointer;
        typedef T&        reference;
        typedef const T&  const_reference;
        typedef ptrdiff_t difference_type;
        //typedef std::forward_iterator_tag iterator_category;

        iterator(): ptr(0) {}
        iterator(node *p): ptr(p) {}

        iterator& operator++() { // preincrement
          ptr = ptr->next;
          return *this;
        }

        iterator operator++(int) { // postincrement
          node *tmp = ptr;
          ptr = ptr->next;
          return iterator(tmp);
        }

        T& operator *() { // dereference
          return ptr->data;
        }

        T* operator ->() {
          return &(ptr->data);
        }

        bool operator == (const iterator &i) { return ptr == i.ptr; }
        bool operator != (const iterator &i) { return ptr != i.ptr; }

    };

    struct const_iterator {
      private:
        const node *ptr;

      public:
        typedef T         value_type;
        typedef T*        pointer;
        typedef T&        reference;
        typedef const T&  const_reference;
        typedef ptrdiff_t difference_type;
        //typedef std::forward_iterator_tag iterator_category;

        const_iterator(): ptr(0) {}
        const_iterator(const node *p): ptr(p) {}
        const_iterator(const iterator &i): ptr(i.ptr) {}

        const_iterator& operator++() { // preincrement
          ptr = ptr->next;
          return *this;
        }

        const_iterator operator++(int) { // postincrement
          const node *tmp = ptr;
          ptr = ptr->next;
          return const_iterator(tmp);
        }

        const T& operator *() { // dereference
          return ptr->data;
        }

        const T* operator ->() {
          return &(ptr->data);
        }

        bool operator == (const const_iterator &i) { return ptr == i.ptr; }
        bool operator != (const const_iterator &i) { return ptr != i.ptr; }

    };

    iterator begin() { return iterator(first); }
    iterator end()   { return iterator(0);  }
    const_iterator begin() const { return const_iterator(first); }
    const_iterator end()   const { return const_iterator(0);  }

    size_type  size()  const { return list_size;  }
    size_type  max_size() const { return size_type(-1); }
    bool empty() const { return (list_size==0); }

    slist() { init_list(); }
    slist(size_type n) {
      init_list();
      for (size_type i=0; i<n; i++)
        push_front(T());  
    }

    slist(size_type n, const_reference t) {
      init_list();
      for (size_type i=0; i<n; i++)
        push_front(t);
    }

    slist(const slist &l){
      init_list();
      for (const_iterator i = l.begin(); i != l.end(); i++)
        push_back(*i);
    }
    
    template <class InputIterator>
    slist(InputIterator f, InputIterator l) {
      init_list();
      for (InputIterator i = f; i != l; i++) {
        push_back(*i);
      }
    }

    ~slist() {
      delete_list();
    }

    slist &operator=(slist &l){
      if (&l != this) {
	delete_list();
	for (iterator i = l.begin(); i != l.end(); i++)
	  push_back(*i);
      }
      return *this;
    }

    void swap(slist &l){
      AprilUtils::swap(first, l.first);
      AprilUtils::swap(last, l.last);
      AprilUtils::swap(list_size, l.list_size);
    }

    // These 2 functions have precondition = !empty()
    reference       front()       { return first->data; }
    const_reference front() const { return first->data; }

    void push_front(const_reference t) {
      first = new node(t, first);
      if (list_size == 0)
        last = first;
      list_size++;
    }

    // Precondition: !empty() 
    void pop_front() {
      node *tmp = first;
      first = first->next;
      delete tmp;
      list_size--;
    }

    void push_back(const_reference t)  {
      if (list_size == 0) {
        first = new node(t,0);
        last = first;
      } else {
        last->next = new node(t, 0);
        last = last->next;
      }
      list_size++;
    }

    // TODO: Reflexionar sobre la conveniencia de meter un metodo
    // que no se parece en nada a lo que hay en la slist de la STL.
    //
    // transfer_front_to_front toma el primer elemento de la lista l y lo inserta
    // en la cabeza de this, sin hacer un push_front y un pop_front que
    // supondrian un new y un delete adicionales.
    void transfer_front_to_front(slist &l) {
      if (l.first) {
        node *aux = l.first;
        // No usamos pop_front porque no queremos hacer un delete del nodo
        l.first = l.first->next;
        l.list_size--;
  
          
        aux->next = first;
        first = aux;
        if (list_size == 0) {
          last = aux;
        }
        list_size++;
      }
    }

    // transfer_front_to_back toma el primer elemento de la lista l y lo inserta
    // en la cola de this, sin hacer un push_back y un pop_front que
    // supondrian un new y un delete adicionales.
    void transfer_front_to_back(slist &l) {
      if (l.first) {
        node *aux = l.first;
        // No usamos pop_front porque no queremos hacer un delete del nodo
        l.first = l.first->next;
        l.list_size--;
	aux->next = 0;
	if (list_size == 0) {
	  last = first = aux; 
	} else {
	  last->next = aux;
	  last = last->next;
	}
	list_size++;
      }
    }

  };
}

#endif

