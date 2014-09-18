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
#ifndef MFSET_H
#define MFSET_H

/* Implementacion de un mfset, permite find(), merge(), toString() y
   fromString() */

extern "C" {
#include <stdint.h>
}

#include "april_assert.h"
#include "constString.h"
#include "binarizer.h"
#include "referenced.h"
#include "maxmin.h"
#include "vector.h"

#define DEFAULT_MFSET_SIZE 524288 // 512*1024

namespace AprilUtils {
  
  class MFSet : public Referenced {
  private:
    // vector que contiene el mfset
    vector<int32_t> data;
    int32_t max_vector_size, vector_size;
  public:
    // constructor de copia
    MFSet(MFSet *obj) {
      this->max_vector_size = obj->max_vector_size;
      this->vector_size     = obj->vector_size;
      this->data.reserve(max_vector_size);
      for (int32_t i=0; i<max_vector_size; ++i)
	this->data[i] = obj->data[i];
    }
    // construye un mefset por defecto donde cada componente de data
    // es un conjunto
    MFSet(int size = DEFAULT_MFSET_SIZE) : data(size) {
      max_vector_size = size;
      data.reserve(max_vector_size);
      vector_size = 0;
      for (int32_t i=0;
	   i<max_vector_size;
	   ++i)
	data[i] = i;
    }
    void setSize(int sz) {
      if (sz >= max_vector_size) {
	int32_t old_max_vector_size = max_vector_size;
	max_vector_size = sz << 1;
	data.resize(max_vector_size);
	for (int32_t i=old_max_vector_size;
	     i<max_vector_size;
	     ++i)
	  data[i] = i;
      }
      vector_size = sz;
    }
    void print() {
      for (int32_t i=0; i<vector_size; ++i)
	printf ("%d ", data[i]);
      printf ("\n");
    }
    // busca un valor en el mfset, y devuelve el conjunto ROOT al que
    // pertence
    int32_t find(const int32_t value) {
      april_assert(value < vector_size);
      int32_t r = data[value];
      while (r != data[r]) {
	data[r] = data[data[r]];
	r = data[r];
      }
      return r;
    }
    // hace un merge de dos conjuntos
    void merge(const int32_t value1,
	       const int32_t value2) {
      if (value1 >= max_vector_size ||
	  value2 >= max_vector_size) {
	int32_t old_max_vector_size = max_vector_size;
	max_vector_size = AprilUtils::max(value1,value2) << 1;
	data.resize(max_vector_size);
	for (int32_t i=old_max_vector_size;
	     i<max_vector_size;
	     ++i)
	  data[i] = i;
      }
      vector_size = AprilUtils::max(value1+1, vector_size);
      vector_size = AprilUtils::max(value2+1, vector_size);
      int32_t r1 = find(value1);
      int32_t r2 = find(value2);
      if (r1 != r2)
	data[r2] = r1;
    }
    // lo borra todo
    void clear() {
      data.clear();
      data.reserve(max_vector_size);
      vector_size = 0;
      for (int32_t i=0;
	   i<max_vector_size;
	   ++i)
	data[i] = i;
    }
    int32_t size() { return vector_size; }
    MFSet *clone() {
      MFSet *nuevo = new MFSet(this);
      return nuevo;
    }
    
    int toString(char **buffer) {
      int32_t sizedata,sizeheader;
      sizeheader = 6;
      sizedata   = binarizer::buffer_size_32(vector_size);
      char *r, *b;
      r = b = new char[sizedata+sizeheader];
      binarizer::code_int32(vector_size, r);
      r += 5;
      *r = '\n';
      ++r;
      r += binarizer::code_vector_int32(data.begin(), vector_size,
					r, sizedata);
      *buffer = b;
      return r-b;
    }

    void fromString(constString cs) {
      cs.extract_int32_binary(&vector_size);
      if (max_vector_size < vector_size) {
	max_vector_size = vector_size*2;
	data.resize(max_vector_size);
	for (int32_t i=0;
	     i<max_vector_size;
	     ++i)
	  data[i] = i;
      }
      for (int i=0; i<vector_size; ++i) {
	int aux;
	cs.extract_int32_binary(&aux);
	data[i] = aux;
      }
    }
  };

} // namespace AprilUtils

#endif //MFSET_H
