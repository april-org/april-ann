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
#ifndef MIN_HEAP_H
#define MIN_HEAP_H

#include "swap.h"

namespace april_utils {

  template<typename T>
  struct CmpLessDefault {
    bool operator()(const T &a, const T &b) {
      return a < b;
    }
  };

  template<typename T,
	   typename CmpLess = CmpLessDefault<T> >
    class min_heap { // minheap binario de valores de tipo T comparables
    CmpLess cmpless;
    T *vec; // vector que empieza con indice 1
    int vector_size;
    int the_size;
    int parent(int pos)     { return pos>>1; }
    int left_child(int pos) { return pos<<1; }
    void heapify(int pos);
    void incr_size();
  public:
    min_heap(int initial_max_size,
	     CmpLess cmpless = CmpLess()) :
      cmpless(cmpless) {
      vector_size = (initial_max_size > 0) ? initial_max_size : 2;
      the_size = 0;
      T *rv = new T[vector_size];
      vec = rv - 1; // vec[1] is rv[0],...
    }
    ~min_heap() { delete[] (vec+1); }
    void clear()       { the_size = 0; }
    bool empty() const { return the_size == 0; }
    int  size()  const { return the_size; }
    const T& top() const { return vec[1]; }
    void pop() {
      if (the_size > 0) {
	vec[1] = vec[the_size];
	the_size--;
	heapify(1);
      }
    }
    void push(const T &elemento) {
      incr_size();
      int position = the_size;
      while ((position > 1) && (cmpless(elemento, vec[parent(position)]))) {
	vec[position] = vec[parent(position)];
	position = parent(position);
      }
      vec[position] = elemento;
    }
    void push_back(const T &elemento) {
      incr_size();
      vec[the_size] = elemento;
    }
    void buildheap() { //  se usa para imponer propiedad de orden
      for (int i = the_size/2; i >= 1; i--) {
	heapify(i);
      }
    }
    
    // ITERADOR
    class iterator {
      friend class min_heap;
      
      // puntero al Heap
      min_heap<T,CmpLess> *h;
      // indice del vector del heap (para iterar)
      int      idx;
    public:
      // constructor...
      iterator(min_heap<T,CmpLess> *p, int i) : h(p), idx(i) { }
      // desreferencia del iterador, devuelve el tipo T
      const T& operator *() const {
	return h->vec[idx];
      }
      // operador flecha para acceder directamente al objeto T
      const T* operator ->() const {
	return &(h->vec[idx]);
      }
      // incremento
      void operator++() { ++idx; }
      // suma
      void operator+(int s) { idx += s; }
      // suma
      void operator+=(int s) { idx += s; }
      // operador de igualdad
      bool operator==(const iterator &b) const {
	return (h   == b.h &&
		idx == b.idx);
      }
      // operador de diferente
      bool operator!=(const iterator &b) const {
	return (h   != b.h ||
		idx != b.idx);
      }
    };
    
    // primer iterador
    iterator begin() {
      return iterator(this, 1);
    }
    // ultimo iterador
    iterator end() {
      return iterator(this, the_size+1);
    }

    // borra una posicion del heap a partir de un iterador situado en la
    // misma
    void del(const min_heap<T,CmpLess>::iterator &idx) {
      // cambiamos el elemento por el ultimo del heap
      vec[idx.idx] = vec[the_size];
      the_size--;
      // hundimos idx
      heapify(idx.idx);
    }
    
  };

  template<typename T,typename CmpLess>
  void min_heap<T,CmpLess>::incr_size() {
    if (the_size >= vector_size) {
      T *old_vec  = vec;
      vec         = new T[2*vector_size] - 1;
      for (int i=1; i<=vector_size; i++)
	vec[i] = old_vec[i];
      vector_size *= 2;
      delete[] (old_vec+1);
    }
    the_size++;
  }

  template<typename T,typename CmpLess>
  void min_heap<T,CmpLess>::heapify(int pos) { // version iterativa
    int child;
    T tmp = vec[pos];
    while (left_child(pos) <= the_size) {
      child = left_child(pos);
      if(child < the_size && cmpless(vec[child+1], vec[child]))
	child++; // el child derecho
      if(cmpless(vec[child], tmp))
	vec[pos] = vec[child];
      else
	break;
      pos = child;
    }
    vec[ pos ] = tmp;
  }

};

#endif // MIN_HEAP_H
