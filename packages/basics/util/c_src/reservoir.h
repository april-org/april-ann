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
#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "slist.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* 

   esta clase sirve para crear un pool de objetos de modo que los
   objetos se sacan del pool con el constructor de copia y se
   devuelven al pool con el destructor

   entre otras cosas, sirve para tener variables caras de construir y
   destruir que se utilicen en un paralel for de openmp usando el
   cualificador firstprivate que llama al constructor de copia y crea
   una instancia por thread en lugar de crear una instancia por
   iteracion

*/

using april_utils::slist;

template <typename T>
struct ReservoirPool {
#ifdef _OPENMP
  omp_lock_t lock;
 #endif
  slist<T*> pool;
  ReservoirPool() {
#ifdef _OPENMP
  omp_init_lock(&lock);
#endif
  }
  ~ReservoirPool() {
#ifdef _OPENMP
  omp_destroy_lock(&lock);
#endif
    while (!pool.empty()) { T *aux = pool.front(); pool.pop_front(); delete aux; }
  }
  T* get() {
    T *resul;
#ifdef _OPENMP
    omp_set_lock(&lock);
#endif
    if (pool.empty()) {
      resul = new T;
    } else {
      resul = pool.front(); pool.pop_front();
    }
#ifdef _OPENMP
  omp_unset_lock(&lock);
#endif
  return resul;
  }
  void release(T *released) {
#ifdef _OPENMP
    omp_set_lock(&lock);
#endif
    pool.push_back(released);
#ifdef _OPENMP
  omp_unset_lock(&lock);
#endif
  }
};

template <typename T>
class ReservoirContainer {
  ReservoirPool<T> *pool;
  T *val;
public:
  T* value() const { return val; }
  ReservoirContainer(ReservoirPool<T> *pool) {
    this->pool = pool;
    val        = pool->get();
  }
  ReservoirContainer(const ReservoirContainer &other) {
    pool = other.pool;
    val  = pool->get();
  }
  ~ReservoirContainer() {
    pool->release(val);
  }
};

template <typename T>
struct VectorReservoirPool {
#ifdef _OPENMP
  omp_lock_t lock;
 #endif
  slist<T*> pool;
  int vector_size;
  VectorReservoirPool(int vector_size) : vector_size(vector_size) {
#ifdef _OPENMP
  omp_init_lock(&lock);
#endif
  }
  ~VectorReservoirPool() {
#ifdef _OPENMP
  omp_destroy_lock(&lock);
#endif
    while (!pool.empty()) { T *aux = pool.front(); pool.pop_front(); delete [] aux; }
  }
  T* get() {
    T *resul;
#ifdef _OPENMP
    omp_set_lock(&lock);
#endif
    if (pool.empty()) {
      resul = new T[vector_size];
    } else {
      resul = pool.front(); pool.pop_front();
    }
#ifdef _OPENMP
  omp_unset_lock(&lock);
#endif
  return resul;
  }
  void release(T *released) {
#ifdef _OPENMP
    omp_set_lock(&lock);
#endif
    pool.push_back(released);
#ifdef _OPENMP
  omp_unset_lock(&lock);
#endif
  }
};

template <typename T>
class VectorReservoirContainer {
  VectorReservoirPool<T> *pool;
  T *val;
public:
  T* value() const { return val; }
  VectorReservoirContainer(VectorReservoirPool<T> *pool) { 
    this->pool = pool;
    val        = pool->get();
  }
  VectorReservoirContainer(const VectorReservoirContainer &other) {
    pool = other.pool;
    val  = pool->get();
  }
  ~VectorReservoirContainer() {
    pool->release(val);
  }
};

#endif // RESERVOIR_H
