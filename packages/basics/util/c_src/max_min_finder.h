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
#ifndef MAX_MIN_FINDER_H
#define MAX_MIN_FINDER_H

#include "vector.h"

namespace AprilUtils {

  // encuentra los maximos/minimos locales de una secuencia, usa contexto
  // izquierdo y derecho, asume que T soporta == y <
  template <typename T> class max_min_finder {
    int ci,cd,maxsz,sz,first;
    T *vec, maxval,minval;
    bool findmax;
    vector<T> *max_output; // puntero a un vector que "nos pasan"
    bool findmin;
    vector<T> *min_output; // puntero a un vector que "nos pasan"
    void process_max_value(T value) { 
      if (max_output) max_output->push_back(value);
    }
    void process_min_value(T value) { 
      if (min_output) min_output->push_back(value);
    }
    T& value_at(int pos) { return vec[(first+pos)%maxsz]; }
    public:
    max_min_finder(int ci=0, int cd=0, 
        bool findmax=true,vector<T>*outvec=0,
        bool findmin=false,vector<T>*invec=0) :
      ci(ci), cd(cd),
      findmax(findmax), max_output(outvec),
      findmin(findmin), min_output(invec) {
        maxsz = ci+1+cd;
        vec = new T[maxsz];
        sz = first = 0;
      }
    ~max_min_finder() { delete[] vec; }
    void put(T entra) {
      if (sz<maxsz) { // estamos llenando el vector
        vec[sz] = entra;
        if (findmax) {
          if (sz == 0 || maxval<entra) maxval=entra;
          if (sz >= cd && vec[sz-cd] == maxval)
            process_max_value(vec[sz-cd]);
        }
        if (findmin) {
          if (sz == 0 || entra<minval) minval=entra;
          if (sz >= cd && vec[sz-cd] == minval)
            process_min_value(vec[sz-cd]);
        }
        sz++;
      } else { // vamos a meter uno nuevo
        T sale = vec[first];
        vec[first] = entra;
        first = (first+1)%maxsz;
        if (findmax) {
          if (maxval < entra) // el nuevo mantiene o mejora el maximo
            maxval = entra;
          else if (sale == maxval) { //  buscar el nuevo maximo
            maxval = vec[first]; // value_at(0)
            for (int i=1;i<maxsz;++i)
              if (maxval<value_at(i)) maxval=value_at(i);
          }
          if (value_at(ci) == maxval) process_max_value(value_at(ci));
        }
        if (findmin) {
          if (entra < minval) // el nuevo mantiene o mejora el minimo
            minval = entra;
          else if (sale == minval) { //  buscar el nuevo minimo
            minval = vec[first]; // value_at(0)
            for (int i=1;i<maxsz;++i)
              if (value_at(i)<minval) minval=value_at(i);
          }
          if (value_at(ci) == minval) process_min_value(value_at(ci));
        }
      }
    }
    void end_sequence() {
      if (sz<maxsz) { // estamos llenando el vector
	int desde = (sz-cd+1 >= 0) ? sz-cd+1 : 0;
        if (findmax)
          for (int i=desde;i<sz;++i)
            if (vec[i] == maxval) process_max_value(vec[i]);
        if (findmin)
          for (int i=desde;i<sz;++i)
            if (vec[i] == minval) process_min_value(vec[i]);
      } else if (cd>0) {
        T aux = value_at(maxsz-1) ;
        for (int i=0; i<cd; ++i) put(aux);
      }
      sz = first = 0;
    }
    void set(int ci=0, int cd=0, 
	     bool findmax=true,vector<T>*out_maxvec=0,
	     bool findmin=false,vector<T>*out_minvec=0) {
      if (maxsz != ci + 1 + cd) {
        delete[] vec;
        maxsz = ci + 1 + cd;
        vec = new T[maxsz];
      }
      this->ci=ci;
      this->cd=cd; 
      this->findmax   =findmax;
      this->max_output=out_maxvec;
      this->findmin   =findmin;
      this->min_output=out_minvec;
      sz = first = 0;
    }
  };

  // encuentra los maximos locales de una secuencia, usa contexto
  // izquierdo y derecho, asume que T soporta == y <
  template <typename T> class max_finder {
    int ci,cd,maxsz,sz,first;
    T *vec, maxval;
    vector<T> *max_output; // puntero a un vector que "nos pasan"
    void process_max_value(T value) { 
      if (max_output) max_output->push_back(value);
    }
    T& value_at(int pos) { return vec[(first+pos)%maxsz]; }
    public:
    max_finder(int ci=0, int cd=0,
		   vector<T>*out_max_vec=0) :
      ci(ci), cd(cd), max_output(out_max_vec) {
        maxsz = ci+1+cd;
        vec = new T[maxsz];
        sz = first = 0;
      }
    ~max_finder() { delete[] vec; }
    void put(T entra) {
      if (sz<maxsz) { // estamos llenando el vector
        vec[sz] = entra;
	if (sz == 0 || maxval<entra) maxval=entra;
	if (sz >= cd && vec[sz-cd] == maxval)
	  process_max_value(vec[sz-cd]);
        sz++;
      } else { // vamos a meter uno nuevo
        T sale = vec[first];
        vec[first] = entra;
        first = (first+1)%maxsz;
	if (maxval < entra) // el nuevo mantiene o mejora el maximo
	  maxval = entra;
	else if (sale == maxval) { //  buscar el nuevo maximo
	  maxval = vec[first]; // value_at(0)
	  for (int i=1;i<maxsz;++i)
	    if (maxval<value_at(i)) maxval=value_at(i);
	}
	if (value_at(ci) == maxval) process_max_value(value_at(ci));
      }
    }
    void end_sequence() {
      if (sz<maxsz) { // estamos llenando el vector
	int desde = (sz-cd+1 >= 0) ? sz-cd+1 : 0;
	for (int i=desde;i<sz;++i)
	  if (vec[i] == maxval) process_max_value(vec[i]);
      } else if (cd>0) {
        T aux = value_at(maxsz-1) ;
        for (int i=0; i<cd; ++i) put(aux);
      }
      sz = first = 0;
    }
    void set(int ci=0, int cd=0, vector<T>*out_max_vec=0) {
      if (maxsz != ci + 1 + cd) {
        delete[] vec;
        maxsz = ci + 1 + cd;
        vec = new T[maxsz];
      }
      this->ci=ci;
      this->cd=cd; 
      this->max_output=out_max_vec;
      sz = first = 0;
    }
  };

  // encuentra los minimos locales de una secuencia, usa contexto
  // izquierdo y derecho, asume que T soporta == y <
  template <typename T> class min_finder {
    int ci,cd,maxsz,sz,first;
    T *vec, minval;
    vector<T> *min_output; // puntero a un vector que "nos pasan"
    void process_min_value(T value) { 
      if (min_output) min_output->push_back(value);
    }
    T& value_at(int pos) { return vec[(first+pos)%maxsz]; }
    public:
    min_finder(int ci=0, int cd=0,
	       vector<T>*out_min_vec=0) :
      ci(ci), cd(cd), min_output(out_min_vec) {
      maxsz = ci+1+cd;
      vec = new T[maxsz];
      sz = first = 0;
    }
    ~min_finder() { delete[] vec; }
    void put(T entra) {
      if (sz<maxsz) { // estamos llenando el vector
        vec[sz] = entra;
	if (sz == 0 || entra<minval) minval=entra;
	if (sz >= cd && vec[sz-cd] == minval)
	  process_min_value(vec[sz-cd]);
        sz++;
      } else { // vamos a meter uno nuevo
        T sale = vec[first];
        vec[first] = entra;
        first = (first+1)%maxsz;
	if (entra < minval) // el nuevo mantiene o mejora el minimo
	  minval = entra;
	else if (sale == minval) { //  buscar el nuevo minimo
	  minval = vec[first]; // value_at(0)
	  for (int i=1;i<maxsz;++i)
	    if (value_at(i)<minval) minval=value_at(i);
	}
	if (value_at(ci) == minval) process_min_value(value_at(ci));
      }
    }
    void end_sequence() {
      if (sz<maxsz) { // estamos llenando el vector
	int desde = (sz-cd+1 >= 0) ? sz-cd+1 : 0;
	for (int i=desde;i<sz;++i)
	  if (vec[i] == minval) process_min_value(vec[i]);
      } else if (cd>0) {
        T aux = value_at(maxsz-1) ;
        for (int i=0; i<cd; ++i) put(aux);
      }
      sz = first = 0;
    }
    void set(int ci=0, int cd=0, vector<T>*out_min_vec=0) {
      if (maxsz != ci + 1 + cd) {
        delete[] vec;
        maxsz = ci + 1 + cd;
        vec = new T[maxsz];
      }
      this->ci=ci;
      this->cd=cd; 
      this->min_output=out_min_vec;
      sz = first = 0;
    }
  };

} // closes namespace AprilUtils

#endif // MAX_MIN_FINDER_H

