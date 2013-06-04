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
#ifndef BINARY_SEARCH_H
#define BINARY_SEARCH_H

/*

Template del algoritmo de busqueda dicotomica

REQUIERE QUE EL TIPO T TENGA DEFINIDA LAS OPERACIONES "<" y "==" con el tipo K

*/

namespace april_utils {

  template<typename T, typename K>
    inline int binary_search(const T *vec, int sz, const K&value) {
    int izq = 0, der = sz-1;
    while (izq <= der) {
      int m = (izq+der)>>1;
      if (vec[m]==value) 
	return m;
      if (vec[m] < value) 
	izq = m+1;
      else 
	der = m-1;
    }
    return -1;
  }

  // REQUIERE QUE EL TIPO T TENGA DEFINIDA LAS OPERACIONES "<" con el tipo K

  template<typename T, typename K>
    inline const T * binary_search_first(const T *vec, int sz, const K&value) {
    int izq = 0, der = sz-2; // OJO: -2, necesita que el vector tenga
			     // un centinela en la posicion sz-1
    while (izq <= der) {
      int m = (izq+der)>>1;
      if (!(vec[m]<value) && vec[m+1]<value)
	return vec+m;
      if (vec[m] < value) 
	izq = m+1;
      else
	der = m-1;
    }
    return 0;
  }

}

#endif //BINARY_SEARCH_H
