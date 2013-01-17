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
#ifndef FIFO_H
#define FIFO_H

namespace april_utils {

  template<typename S>
    struct fifo_node {
      S value;
      fifo_node<S> *next;
    fifo_node(S val) : value(val), next(0) { }
    };

  template<typename T>
    class fifo {
    fifo_node<T> *first, *last;
    int the_size;
  public:
    fifo();
    ~fifo();
    const fifo_node<T> *begin() const { return first; }
    //const fifo_node<T> *end()   const { return last;  }
    int size() const;
    bool empty() const;
    void put(T value);
    bool get(T &value);
    bool consult(T &value);
    bool drop_by_value(T value);
  };

  template<typename T>
    fifo<T>::fifo() :
  first(0), last(0), the_size(0)
    {
    }

  template<typename T>
    fifo<T>::~fifo() {
    while (first) {
      fifo_node<T> *aux = first;
      first = first->next;
      // TODO ¿delete value? si:
      // delete aux->value;
      delete aux;
    }
  }

  template<typename T>
    int fifo<T>::size() const {
    return the_size;
  }

  template<typename T>
    bool fifo<T>::empty() const {
    return the_size == 0;
  }
 
  template<typename T>
    void fifo<T>::put(T value) {
    fifo_node<T> *aux = new fifo_node<T>(value);
    if (first)
      last->next = aux;
    else
      first = aux;
    last = aux;
    the_size++;
  }

  template<typename T>
    bool fifo<T>::get(T &value) {
    if (!first) return false;
    the_size--;
    value = first->value;
    if (first == last) last = 0;
    fifo_node<T> *aux = first;
    first = first->next;
    delete aux;
    return true;
  }

  template<typename T>
    bool fifo<T>::consult(T &value) {
    if (!first) return false;
    value = first->value;
    return true;
  }

  template<typename T>
    bool fifo<T>::drop_by_value(T value) {
    fifo_node<T> *prev = 0;
    fifo_node<T> *reco = first;
    while (reco && reco->value != value) {
      prev = reco;
      reco = reco->next;
    }
    if (reco == 0) 
      return false;
    // eliminar valor apuntado por reco
    the_size--;
    if (last == reco)
      last = prev;
    if (prev == 0) {
      first = reco->next;
    } else {
      prev->next = reco->next;
    }
    delete reco;
    //
    return true;
  }

}

#endif // FIFO_H
