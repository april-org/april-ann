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
#ifndef QSORT_H
#define QSORT_H

#include "swap.h"

namespace april_utils {

  // use T::operator<()
  template<typename T>
  inline void InsertionSort(T v[], int low, int high) {
    int i,j;
    T aux;
    for (i = low+1; i <= high; i++) {
      aux=v[i];
      for (j = i-1; j>=low && aux<v[j]; j--)
        v[j+1]=v[j];
      v[j+1]=aux;
    }
  }

  // use specified comparison function
  template<typename T, typename Compare>
  inline void InsertionSort(T v[], int low, int high, Compare compare) {
    int i,j;
    T aux;
    for (i = low+1; i <= high; i++) {
      aux=v[i];
      for (j = i-1; j>=low && compare(aux,v[j]); j--)
        v[j+1]=v[j];
      v[j+1]=aux;
    }
  }

  // use T::operator<()
  template<typename T>
  inline T med3(T v[], int low, int high) {
    int cnt = (low+high)/2;
    if (v[cnt] <v[low]) swap(v[low],v[cnt]);
    if (v[high]<v[low]) swap(v[low],v[high]);
    if (v[high]<v[cnt]) swap(v[cnt],v[high]);
    swap(v[cnt],v[high-1]);
    return(v[high-1]);
  }

  // use specified comparison function
  template<typename T, typename Compare>
  inline T med3(T v[], int low, int high, Compare compare) {
    int cnt = (low+high)/2;
    if (compare(v[cnt] ,v[low])) swap(v[low],v[cnt]);
    if (compare(v[high],v[low])) swap(v[low],v[high]);
    if (compare(v[high],v[cnt])) swap(v[cnt],v[high]);
    swap(v[cnt],v[high-1]);
    return(v[high-1]);
  }

  // use T::operator<()
  template<typename T>
  inline int partition(T v[], int low, int high) {
    // precondition: low+1 < high
    int izq,der;
    T piv = med3(v,low,high);
    izq=low;der=high-1;
    do {
      do {
        izq++;
      } while(v[izq]<piv);
      do {
        der--;
      }  while(piv<v[der]);
      swap(v[izq],v[der]);
    } while (izq<der);
    swap(v[izq],v[der]);
    swap(v[izq],v[high-1]);
    return izq;
  }

  // use specified comparison function
  template<typename T, typename Compare>
  inline int partition(T v[], int low, int high, Compare compare) {
    // precondition: low+1 < high
    int izq,der;
    T piv = med3(v,low,high, compare);
    izq=low;der=high-1;
    do {
      do {
        izq++;
      } while(compare(v[izq],piv));
      do {
        der--;
      }  while(compare(piv,v[der]));
      swap(v[izq],v[der]);
    } while (izq<der);
    swap(v[izq],v[der]);
    swap(v[izq],v[high-1]);
    return izq;
  }


  // use T::operator<()
  template<typename T>
  void Sort(T v[], int low, int high) {
    const int MIN_SIZE = 8;
    // Stack space optimization: the smaller partition is dealt with recursively,
    // and then the bigger one is sorted by tail recursion/iteration, this way
    // the space complexity is theta(log n), even for the worst case
    // Also, halt recursion and iteration when the section of the
    // array to be sorted is less than MIN_SIZE long, and use InsertionSort instead
    while (high - low >= MIN_SIZE) {
      int p = partition(v, low, high);

      if (p - low < high - p) {
        Sort(v, low, p);
        low = p + 1;
      } else {
        Sort(v, p + 1, high);
        high = p;
      }
    }
    InsertionSort(v, low, high);
  }

  // use specified comparison function
  template<typename T, typename Compare>
  void Sort(T v[], int low, int high, Compare compare) {
    const int MIN_SIZE = 8;
    // Stack space optimization: the smaller partition is dealt with recursively,
    // and then the bigger one is sorted by tail recursion/iteration, this way
    // the space complexity is theta(log n), even for the worst case
    // Also, halt recursion and iteration when the section of the
    // array to be sorted is less than MIN_SIZE long, and use InsertionSort instead
    while (high - low >= MIN_SIZE) {
      int p = partition(v, low, high, compare);

      if (p - low < high - p) {
        Sort(v, low, p, compare);
        low = p + 1;
      } else {
        Sort(v, p + 1, high, compare);
        high = p;
      }
    }
    InsertionSort(v, low, high, compare);
  }


  /**

     Template del algoritmo quicksort en la version que realiza una
     sola llamada recursiva (para evitar que en el caso peor la pila
     crezca de forma lineal con la talla del vector. En un caso base
     inferior a MINSIZE (ahora es constante, TODO: se puede meter como
     un valor en el template y dar valor por defecto) se utiliza
     InsertionSort.

     REQUIERE QUE EL TIPO T TENGA DEFINIDA LA OPERACION DE COPIA Y LA
     COMPARACION MENOR ESTRICTO "<".

     Se utiliza basicamente asi:

     \code
     T v[n];
     Sort(v,n); // ordenamos el vector
     \endcode

  */
  template<typename T>
  void Sort(T v[], int sze) {
    Sort(v,0,sze-1);
  }

  template<typename T, typename Compare>
  void Sort(T v[], int sze, Compare compare) {
    Sort(v,0,sze-1, compare);
  }

  template<typename T>
  T Selection(T v[], int sze, int k) {
    const int MIN_SIZE = 8;
    int low=0,high=sze-1,q;
    while (high - low >= MIN_SIZE) {
      q = partition(v, low, high);
      if (k <= q)
        high = q;
      else
        low  = q+1;
    }
    InsertionSort(v, low, high);
    return v[k];
  }

  template<typename T, typename Compare>
  T Selection(T v[], int sze, int k, Compare cmp) {
    const int MIN_SIZE = 8;
    int low=0,high=sze-1,q;
    while (high - low >= MIN_SIZE) {
      q = partition(v, low, high, cmp);
      if (k <= q)
        high = q;
      else
        low  = q+1;
    }
    InsertionSort(v, low, high, cmp);
    return v[k];
  }

  template<typename T>
  class PartQ { // QuickSortQueue
    static const int threshold =  15;
    static const int sizestack = 200;
    T*  v;
    int size;
    int first;
    int stack[sizestack];
    int topStack;
    int lastSorted;
    void push(int x) { topStack++; stack[topStack] = x; }
    int top()  { return stack[topStack]; }
    void pop() { topStack--; }
    void increase_lastSorted() {
      while (top() < lastSorted)
        pop();
      int ini = lastSorted+1;
      while (top()-ini >= threshold) {
        push(partition(v, ini, top()-1));
      }
      InsertionSort(v,ini,top()-1);
      lastSorted = top(); pop();
    }
  public:
    PartQ()  {}
    ~PartQ() {}
    void configure(T* vector, int size) {
      v = vector;
      this->size = size;
      first = 0;
      lastSorted = -1;
      topStack = -1;
      push(size);
    }
    bool extractMin(T& value) {
      if (first >= size)
        return false;
      if (first > lastSorted) {
        increase_lastSorted();
      }
      value = v[first];
      first++;
      return true;
    }
    void sort_up_to(int index) {
      if (index >= size) index = size-1;
      while (lastSorted < index)
        increase_lastSorted();
    }
    int getSize() { return size - first; }
  };

} // namespace

#endif //QSORT_H
