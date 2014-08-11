/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Francisco Zamora-Martinez, Jorge
 * Gorbe-Moya
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
#ifndef DATASET_CC_H
#define DATASET_CC_H

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "dataset.h"
#include "mod.h"
#include "clamp.h"
#include "error_print.h"
#include "maxmin.h"
#include "april_assert.h"

using april_utils::mod;
using april_utils::clamp;

// ---------------------------------------------------------------------

template <typename T>
MatrixDataSet<T>::MatrixDataSet(Matrix<T> *m) : DataSet<T>() {
  if (!m->isSimple())
    ERROR_EXIT(128, "Matrix need to be simple (contiguous "
	       "and in row-major)\n");
  // numdim debe ser siempre >0
  int d        = m->getNumDim();	
  matrix       = m;
  IncRef(matrix); // garbage collection
  offset       = new int[d];
  subMatrixSize= new int[d];
  circular     = new bool[d];
  step         = new int[d];
  numSteps     = new int[d];
  orderStep    = new int[d];
  coordinate   = new int[d];
  // valores por defecto
  defaultValue = 0;
  for (int i=0; i<d; i++) {
    offset[i]        = 0;
    subMatrixSize[i] = m->getDimSize(i); // ocupa todo el ancho
    orderStep[i]     = d-(i+1);
    step[i]          = 1;
    numSteps[i]      = 1 ; // m->getDimSize(i);
    circular[i]      = false;
  }
  if (d > 0) {
    subMatrixSize[0] = 1;
    numSteps[0]      = m->getDimSize(0);
  }
  // calcular estos otros valores:
  numPatternsv = patternSizev = 1;
  for(int i=0;i<d;i++) {
    numPatternsv *= numSteps[i];
    patternSizev *= subMatrixSize[i];
  }
}

template <typename T>
void MatrixDataSet<T>::setValue(int *dest, int *orig){
  for(int i=0;i<matrix->getNumDim();i++){
    dest[i]=orig[i];
  }
}

template <typename T>
void MatrixDataSet<T>::setValue(bool *dest, bool *orig){
  for(int i=0;i<matrix->getNumDim();i++){
    dest[i]=orig[i];
  }
}

template <typename T>
void MatrixDataSet<T>::setNumSteps(int *v) {
  setValue(numSteps,v);
  numPatternsv=1;
  for(int i=0;i<matrix->getNumDim();i++) {
    if (numSteps[i] == 0)
      ERROR_EXIT1(1, "Error: numsteps[%d] == 0\n",i);
    numPatternsv=numPatternsv*numSteps[i];
  }
}

template <typename T>
void MatrixDataSet<T>::setSubMatrixSize(int *v) {
  setValue(subMatrixSize,v);
  patternSizev=1;
  for(int i=0;i<matrix->getNumDim();i++)	
    patternSizev=patternSizev*subMatrixSize[i];
}

template <typename T>
void MatrixDataSet<T>::index2coordinate(int index) {
  for (int i=0; i < matrix->getNumDim();i++) {
    int j = orderStep[i];
    coordinate[j] = offset[j] + (index % numSteps[j])*step[j];
    index=index/numSteps[j];
  }
}

template <typename T>
void MatrixDataSet<T>::auxGetPattern(int offsetmatrix, int d) {
  int i,c,t;
  t = matrix->getDimSize(d);
  // recursiva
  if (d == matrix->getNumDim()-1) {
    // ultima dimension, caso base
    const T *data = matrix->getRawDataAccess()->getPPALForRead() + offsetmatrix*t;
    if (circular[d]) {
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	pattern[offsetpat++] = data[mod(c,t)];
      }
    } else {
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	pattern[offsetpat++] = 
	  ((c < 0 || c >= t) ? defaultValue : data[c]);
      }
    }
  } else { // no es ultima dimension
    if (circular[d]) {
      // caso no base, es circular
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	auxGetPattern(offsetmatrix*matrix->getDimSize(d+1)+mod(c,t),d+1);
      } // cierra for caso base no circular
    } else { // caso no base, no circular
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	if (c < 0 || c >= t) {
	  // rellenar el resto de dimensiones con defaultValue
	  int veces = 1; 
	  for (int j = d+1; j < matrix->getNumDim(); j++)
	    veces *= subMatrixSize[j];
	  for (;veces;veces--)
	    pattern[offsetpat++] = defaultValue;
	} else {
	  auxGetPattern(offsetmatrix*matrix->getDimSize(d)+c,d+1);
	}
      } // cierra for : caso base no circular
    } // cierra else : caso no base, no circular
  } // cierra else : no es ultima dimension
}

//se asume que T *pat ya es un vector suficientemente grande
template <typename T>
int MatrixDataSet<T>::getPattern(int index, T *pat){
  index2coordinate(index); // actualiza vector coordinate
  pattern = pat;
  offsetpat = 0;
  auxGetPattern(0, 0); // recibe offsetmatrix y dimension a tratar
  return patternSize();
}

template <typename T>
void MatrixDataSet<T>::auxPutPattern(int offsetmatrix, int d) {
  int i,c,t;
  t = matrix->getDimSize(d);
  // recursiva
  if (d == matrix->getNumDim()-1) {
    // ultima dimension, caso base
    T *data = matrix->getRawDataAccess()->getPPALForWrite() + offsetmatrix*t;    
    if (circular[d]) {
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	data[mod(c,t)] = const_pattern[offsetpat++];
      }
    } else {
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	if ((0 <= c) && (c < t)) {
	  data[c] = const_pattern[offsetpat];
	}
	offsetpat++;
      }
    }
  } else { // no es ultima dimension
    if (circular[d]) {
      // caso no base, es circular
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	auxPutPattern(offsetmatrix*matrix->getDimSize(d+1)+(mod(c,t)),d+1);
      } // cierra for caso base no circular
    } else { // caso no base, no circular
      for(i = subMatrixSize[d], c = coordinate[d]; i; c++,i--) {
	if (c < 0 || c >= t) { // te sales de la matriz
	  // saltarse el resto de dimensiones
	  int veces = 1; 
	  for (int j = d+1; j < matrix->getNumDim(); j++)
	    veces *= subMatrixSize[j];
	  offsetpat += veces;
	} else {
	  auxPutPattern(offsetmatrix*matrix->getDimSize(d)+c,d+1);
	}
      } // cierra for : caso base no circular
    } // cierra else : caso no base, no circular
  } // cierra else : no es ultima dimension
}

//se asume que T *pat ya es un vector suficientemente grande
template <typename T>
int MatrixDataSet<T>::putPattern(int index, const T *pat) {
  index2coordinate(index); // actualiza vector coordinate
  const_pattern = pat; // pattern es atributo que se va modificando
  offsetpat = 0;   // idem
  auxPutPattern(0, 0); // llamada recursiva inicial
  return patternSize();
}

template <typename T>
MatrixDataSet<T>::~MatrixDataSet(){
  delete[] offset;
  delete[] subMatrixSize;
  delete[] circular;
  delete[] step;
  delete[] numSteps;
  delete[] coordinate;
  delete[] orderStep;
  DecRef(matrix); // garbage collection
}

// ---------------------------------------------------------------------

template <typename T>
IdentityDataSet<T>::IdentityDataSet(int patternSize,
				    T zerovalue,
				    T onevalue) : DataSet<T>(),
                                                  zerovalue(zerovalue),
						  onevalue(onevalue) {
  patternsz = patternSize;
}
  
template <typename T>
IdentityDataSet<T>::~IdentityDataSet() {
  // Nothing to do
}
  
template <typename T>
int IdentityDataSet<T>::getPattern(int index, T *pat) {
  april_assert("Incorrect index" && index >= 0 && index < numPatterns());
  for (int i=0; i<patternsz; i++)
    pat[i] = zerovalue;
  pat[index] = onevalue;
  return patternsz;
}

template <typename T>
int IdentityDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  return 0;
}

// ---------------------------------------------------------------------

template <typename T>
UnionDataSet<T>::UnionDataSet(int n, DataSet<T> **v) : DataSet<T>() {
  num = n;
  vds = new DataSet<T>*[num];
  d = new int[num+1];
  d[0] = 0;
  patternsz = v[0]->patternSize();
  int i;
  for (i=0; i < num; i++) {
    vds[i] = v[i];
    IncRef(v[i]); // garbage collection
    if (patternsz != v[i]->patternSize()) {
      // ERROR!!!!
      // todo: comprobar esto en el binding
      ERROR_EXIT(1,"ERROR dataset.union: one of datasets has wrong patternsize\n");
    }
    d[i+1] = d[i] + v[i]->numPatterns();
  }
}

template <typename T>
UnionDataSet<T>::~UnionDataSet() {
  for (int i=0; i < num; i++) {
    DecRef(vds[i]); // garbage collection
  }
  delete[] vds; // necesitamos borrar tambien los datasets apuntados?
  delete[] d;
}

template <typename T>
int UnionDataSet<T>::getPattern(int index, T *pat) {
  // buscamos el dataset asociado y su indice correspondiente
  if (index < 0 || index >= numPatterns())
    return 0;
  int izq,der,m;
  izq = 0; der = num;
  do {
    m = (izq+der)/2;
    if (d[m] <= index) 
      izq = m; 
    else 
      der = m;
  } while (izq < der-1);
  return vds[izq]->getPattern(index-d[izq],pat);
};

template <typename T>
int UnionDataSet<T>::putPattern(int index, const T *pat) {
  // buscamos el dataset asociado y su índice correspondiente
  if (index < 0 || index >= numPatterns())
    return 0;
  int izq,der,m;
  izq = 0; der = num;
  do {
    m = (izq+der)/2;
    if (d[m] <= index) 
      izq = m; 
    else 
      der = m;
  } while (izq < der-1);
  return vds[izq]->putPattern(index-d[izq],pat);
};

// ---------------------------------------------------------------------

template <typename T>
SubDataSet<T>::SubDataSet(int ini, int fin, DataSet<T> *ds) : DataSet<T>() {
  this->ds = ds;
  IncRef(ds); // garbage collection
  this->ini = ini;
  this->fin = fin;
  this->size = fin - ini + 1;
}

template <typename T>
SubDataSet<T>::~SubDataSet() {
  DecRef(ds); // garbage collection
}

template <typename T>
int SubDataSet<T>::getPattern(int index, T *pat) {
  // TODO: falta comprobar que el indice esta en el rango correcto
  return ds->getPattern(ini + index,pat);
}

template <typename T>
int SubDataSet<T>::putPattern(int index, const T *pat) {
  // TODO: falta comprobar que el indice esta en el rango correcto
  return ds->putPattern(ini + index,pat);
}

// ---------------------------------------------------------------------

template <typename T>
SplitDataSet<T>::SplitDataSet(int ini, int fin, DataSet<T> *ds) : DataSet<T>() {
  this->ini = ini;
  this->fin = fin;
  this->ds = ds;
  IncRef(ds); // garbage collection
  size = fin-ini+1;
  aux = new T[ds->patternSize()];
}

template <typename T>
SplitDataSet<T>::~SplitDataSet() {
  DecRef(ds); // garbage collection
  delete[] aux;
}

template <typename T>
int SplitDataSet<T>::getPattern(int index, T *pat) {
  ds->getPattern(index,aux);
  T *r = aux+ini;
  for (int i = 0; i < size; i++)
    pat[i] = r[i];
  return size;
}

// ---------------------------------------------------------------------

template <typename T>
JoinDataSet<T>::JoinDataSet(int n, DataSet<T> **v) : DataSet<T>() {
  num = n;
  vds = new DataSet<T>*[num];
  d   = new int[num+1];
  d[0]= 0;
  int ps = v[0]->numPatterns();
  int i;
  for (i=0; i < num; i++) {
    vds[i] = v[i];
    IncRef(v[i]); // garbage collection
    if (ps != v[i]->numPatterns()) {
      // ERROR!!!!
      ERROR_EXIT(1,"Error in JoinDataSet constructor\n");
    }
    d[i+1] = d[i] + v[i]->patternSize();
  }
}

template <typename T>
JoinDataSet<T>::~JoinDataSet() {
  for (int i=0; i < num; i++) {
    DecRef(vds[i]); // garbage collection
  }
  delete[] vds;
  delete[] d;
}

template <typename T>
int JoinDataSet<T>::getPattern(int index, T *pat) {
  for (int i=0; i < num; i++) 
    vds[i]->getPattern(index, pat+d[i]);
  return patternSize();
}

template <typename T>
int JoinDataSet<T>::putPattern(int index, const T *pat) {
  for (int i=0; i < num; i++) 
    vds[i]->putPattern(index, pat+d[i]);
  return patternSize();
}

// ---------------------------------------------------------------------

template <typename T>
IndexDataSet<T>::IndexDataSet(DataSet<T> **datasets, int firstindex) :
  DataSet<T>() {
  this->firstindex = firstindex;
  indices = datasets[0];
  IncRef(indices); // garbage collection
  numdiccionarios = indices->patternSize();
  diccionarios = new DataSet<T>*[numdiccionarios];
  patternindices = new T[numdiccionarios];
  if (numdiccionarios == 0) {
    ERROR_EXIT(1,"error: indexDataSet needs at least 1 dictionary\n");
  }
  patternsize = 0;
  for (int i=0; i < numdiccionarios; i++) {
    diccionarios[i] = datasets[i+1];
    IncRef(diccionarios[i]); // garbage collection
    patternsize +=   diccionarios[i]->patternSize();
  }
}

template <typename T>
IndexDataSet<T>::~IndexDataSet() {
  delete[] patternindices;
  DecRef(indices); // garbage collection
  for (int i=0; i < numdiccionarios; i++) {
    DecRef(diccionarios[i]); // garbage collection
  }
  delete[] diccionarios;
}

template <typename T>
int IndexDataSet<T>::getPattern(int index, T *pat) {
  int pos = 0;
  indices->getPattern(index,patternindices);
  for (int i=0; i < numdiccionarios; i++) {
    int idx = static_cast<int>(patternindices[i])-firstindex;
    april_assert("Incorrect index at IndexDataSet" && idx >= 0);
    pos += diccionarios[i]->getPattern((int)idx,pat+pos);
  }
  return patternSize();
}

template <typename T>
int IndexDataSet<T>::putPattern(int index, const T *pat) {
  int pos = 0;
  index += firstindex;
  indices->getPattern(index,patternindices);
  for (int i=0; i < numdiccionarios; i++) {
    pos += diccionarios[i]->putPattern((int)patternindices[i]-firstindex,pat+pos);
  }
  return patternSize();
}

// ---------------------------------------------------------------------

template <typename T>
LinearCombDataSet<T>::LinearCombDataSet(DataSet<T> *ds,
                                        LinearCombConf<T> *conf) :
  DataSet<T>() {
  this->ds = ds;
  aux  = new T[ds->patternSize()+1]; aux[0] = 1;
  IncRef(ds); // garbage collection
  this->conf = conf;
  IncRef(conf); // garbage collection
}

template <typename T>
LinearCombDataSet<T>::~LinearCombDataSet() {
  delete[] aux;
  DecRef(conf); // garbage collection
  DecRef(ds);   // garbage collection
}

template <typename T>
int LinearCombDataSet<T>::getPattern(int index, T *pat) {
  int i,j,desde,hasta;
  desde = 0;
  ds->getPattern(index,aux+1);
  for (i=0; i<conf->patternsize; i++) { // recorremos indices de salida
    pat[i] = 0;
    hasta = conf->numTuplas[i];
    for (j=desde;j<hasta;j++)
      pat[i] += aux[conf->indices[j]]*conf->pesos[j];
    desde = hasta;
  }
  return conf->patternsize;
}

// ---------------------------------------------------------------------

template <typename T>
ContextualizerDataSet<T>::ContextualizerDataSet(DataSet<T> *ds,
						int izq, int der,
						bool reverse) :
  DataSet<T>(),
  ctxtizq(izq), ctxtder(der), ds(ds), reverse(reverse) {
  IncRef(ds); // garbage collection
  numpatterns = ds->numPatterns();
  patternsize = ds->patternSize()*(ctxtizq+1+ctxtder);
}

template <typename T>
ContextualizerDataSet<T>::~ContextualizerDataSet() {
  DecRef(ds); // garbage collection
}

template <typename T>
int ContextualizerDataSet<T>::getPattern(int index, T *pat) {
  T *vec = pat;
  int ps = ds->patternSize(), i,j = index-ctxtizq;
  if (reverse) {
    vec += (ctxtizq+ctxtder)*ps;
    ps   = -ps;
  }
  for (i=0; i<ctxtizq; i++,vec += ps, j++)
    ds->getPattern(((j >= 0) ? j : 0), vec);
  ds->getPattern(j, vec);
  vec += ps; j++;
  for (i=0; i<ctxtder; i++,vec += ps, j++)
    ds->getPattern(((j < numpatterns) ? j : numpatterns-1), vec);
  return patternsize;
}

template <typename T>
int ContextualizerDataSet<T>::putPattern(int index, const T *pat) {
  const T *vec = pat;
  int ps = ds->patternSize(), i,j = index-ctxtizq;
  for (i=0; i<ctxtizq; i++,vec += ps, j++)
    if (j >= 0)
      ds->putPattern(j, vec);
  ds->putPattern(j, vec);
  vec += ps; j++;
  for (i=0; i<ctxtder; i++,vec += ps, j++)
    if (j < numpatterns)
      ds->putPattern(j, vec);
  return patternsize;
}

// ---------------------------------------------------------------------

template <typename T>
AccumulateDataSet<T>::AccumulateDataSet(int patsz, int numpat) : DataSet<T>() {
  patternsize = patsz;
  numpatterns = numpat;
  data        = new double[patsz];
  for (int i=0;i<patsz;i++) 
    data[i] = 0.0;
}
 
template <typename T>
AccumulateDataSet<T>::~AccumulateDataSet() {
  delete[] data;
}

template <typename T>
int AccumulateDataSet<T>::getPattern(int index, T *pat) {
  UNUSED_VARIABLE(index);
  //   if (index < 0 || index >= numpatterns)
  //     return 0;
  for (int i=0; i<patternsize; i++)
    pat[i] = (T)data[i];
  return patternsize;
}

template <typename T>
int AccumulateDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  //   if (index < 0 || index >= numpatterns)
  //     return 0;
  for (int i=0; i<patternsize; i++)
    data[i] += (double)pat[i];
  return patternsize;
}

// ---------------------------------------------------------------------

template <typename T>
ByteDataSet<T>::ByteDataSet(int patsz, int numpat,
			    double a, double b) : DataSet<T>() {
  patternsize = patsz;
  numpatterns = numpat;
  this->a     = a;
  this->b     = b;
  data        = new unsigned char[patsz*numpat];
  for (int i=0;i<patsz*numpat;i++)
    data[i] = (unsigned char)0;
}
 
template <typename T>
ByteDataSet<T>::~ByteDataSet() {
  delete[] data;
}

template <typename T>
int ByteDataSet<T>::getPattern(int index, T *pat) {
  //   if (index < 0 || index >= numpatterns)
  //     return 0;
  int offset = index*patternsize;
  for (int i=0; i<patternsize; i++)
    pat[i] = (T)(a*data[offset+i]+b);
  return patternsize;
}

template <typename T>
int ByteDataSet<T>::putPattern(int index, const T *pat) {
  //   if (index < 0 || index >= numpatterns)
  //     return 0;
  int offset = index*patternsize;
  for (int i=0; i<patternsize; i++)
    data[offset+i] = (unsigned char)clamp((int)round((pat[i]-b)/a),0,255);
  return patternsize;
}

// ---------------------------------------------------------------------


template <typename T>
BitDataSet<T>::BitDataSet(int nump, int patsize) : DataSet<T>() {
  numpatterns     = nump;
  patternsize     = patsize;
  int p           = ((numpatterns*patternsize)>>3) + 1;
  data            = new unsigned char[p];
  for (int i=0; i<p; ++i)
    data[i] = 0;
}

template <typename T>
BitDataSet<T>::~BitDataSet() {
  delete[] data;
}

template <typename T>
int BitDataSet<T>::getPattern(int index, T *pat) {
  index       = index * patternsize;
  int bytePos = index >> 3;
  int bitPos  = index & (0x07);
  for (int i=0; i<patternsize; ++i) {
    unsigned char v = (data[bytePos] >> bitPos) & (0x01);
    if (v) pat[i] = T(1);
    else pat[i] = T(0);
    ++bitPos;
    if (bitPos > 7) {
      bitPos = 0;
      bytePos++;
    }
  }
  return patternsize;
}

template <typename T>
int BitDataSet<T>::putPattern(int index, const T *pat) {
  index       = index * patternsize;
  int bytePos = index >> 3;
  int bitPos  = index & (0x07);
  for (int i=0; i<patternsize; ++i) {
    if (pat[i] == T(0))
      data[bytePos] = data[bytePos] & (0xff ^ (1 << bitPos));
    else data[bytePos] = data[bytePos] | (1 << bitPos);
    ++bitPos;
    if (bitPos > 7) {
      bitPos = 0;
      bytePos++;
    }
  }
  return patternsize;
}

// ---------------------------------------------------------------------

template <typename T>
SparseMatrixDataset<T>::SparseMatrixDataset(SparseMatrix<T> *m) :
  DataSet<T>(),
  matrix(m), numpatterns(m->getDimSize(0)), patternsize(m->getDimSize(1)) {
  if (m->getSparseFormat() != CSR_FORMAT) {
    ERROR_EXIT(128, "SparseMatrix need to be in CSR format\n");
  }
  IncRef(m);
}

template <typename T>
SparseMatrixDataset<T>::~SparseMatrixDataset() {
  DecRef(matrix);
}

template <typename T>
int SparseMatrixDataset<T>::getPattern(int index, T *pat) {
  // WARNING: only works with float
  for (int i=0; i<patternsize; ++i) pat[i] = T(0.0f);
  typename SparseMatrix<T>::const_iterator it(matrix->iteratorAtFirst(index,0));
  while(true) {
    int x0,x1;
    it.getCoords(x0,x1);
    if (x0 != index) break;
    pat[x1] = *it;
  }
  return patternsize;
}

template <typename T>
int SparseMatrixDataset<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for SparseMatrixDataset!!!\n");
  return 0;
}

// ---------------------------------------------------------------------

template <typename T>
ShortListDataSet<T>::ShortListDataSet(DataSet<T> *ds, int short_list_size,
                                      int unk_word) :
  DataSet<T>(),
  ds(ds), short_list_size(short_list_size), unk_word(unk_word) {
  IncRef(ds);
  patsize = ds->patternSize();
}

template<typename T>
ShortListDataSet<T>::~ShortListDataSet() {
  DecRef(ds);
}

template<typename T>
int ShortListDataSet<T>::getPattern(int index, T *pat) {
  UNUSED_VARIABLE(index);
  //int aux = ds->getPattern(index, pat);
  for (int i=0; i<patsize; ++i)
    if (pat[i] > short_list_size) pat[i] = unk_word;
  return patsize;
}

template<typename T>
int ShortListDataSet<T>::putPattern(int index, const T *pat) {
  return ds->putPattern(index, pat);
}


//----------------------------------------------------------

template <typename T>
IndexFilterDataSet<T>::IndexFilterDataSet(DataSet<T> *ds,
					  ReferencedVectorUint *indexes) :
  DataSet<T>(),
  ds(ds), indexes(indexes) {
  IncRef(ds);
  IncRef(indexes);
}

template<typename T>
IndexFilterDataSet<T>::~IndexFilterDataSet() {
  DecRef(ds);
  DecRef(indexes);
}

template<typename T>
int IndexFilterDataSet<T>::getPattern(int index, T *pat) {
  // OJO: -1 porque la tabla viene de LUA y comienza en 1
  return ds->getPattern((*indexes)[index]-1, pat);
}

template<typename T>
int IndexFilterDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for IndexFilterDataSet!!!\n");
  return 0;  
}

//----------------------------------------------------------

template <typename T>
PerturbationDataSet<T>::PerturbationDataSet(DataSet<T> *ds, MTRand *random,
					    double mean,
					    double variance) :
  DataSet<T>(),
  ds(ds), random(random), mean(mean), variance(variance) {
  IncRef(ds);
  IncRef(random);
}

template<typename T>
PerturbationDataSet<T>::~PerturbationDataSet() {
  DecRef(ds);
  DecRef(random);
}

template<typename T>
int PerturbationDataSet<T>::getPattern(int index, T *pat) {
  int ret = ds->getPattern(index, pat);
  int sz  = ds->patternSize();
  for (int i=0; i<sz; ++i)
    pat[i] += random->randNorm(mean,variance);
  return ret;
}

template<typename T>
int PerturbationDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for PerturbationDataSet!!!\n");
  return 0;  
}

//-------------------------------------------------------------

template <typename T>
SaltNoiseDataSet<T>::SaltNoiseDataSet(DataSet<T> *ds, MTRand *random,
				      double vd,
				      T zero) :
  DataSet<T>(),
  ds(ds), random(random), vd(vd), zero(zero),
  number_of_zeroes(static_cast<int>(vd*ds->patternSize())) {
  IncRef(ds);
  IncRef(random);
  zero_positions = new int[ds->patternSize()];
}

template<typename T>
SaltNoiseDataSet<T>::~SaltNoiseDataSet() {
  DecRef(ds);
  DecRef(random);
  delete[] zero_positions;
}

template<typename T>
int SaltNoiseDataSet<T>::getPattern(int index, T *pat) {
  int ret = ds->getPattern(index, pat);
  int sz  = ds->patternSize();
  random->shuffle(sz, zero_positions);
  for (int i=0; i<number_of_zeroes; ++i) pat[zero_positions[i]] = zero;
  return ret;
}

template<typename T>
int SaltNoiseDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for SaltNoiseDataSet!!!\n");
  return 0;  
}

//-------------------------------------------------------------

template <typename T>
SaltPepperNoiseDataSet<T>::SaltPepperNoiseDataSet(DataSet<T> *ds, MTRand *random,
						  double vd,
						  T zero,
						  T one) :
  DataSet<T>(),
  ds(ds), random(random), vd(vd), zero(zero), one(one),
  number_of_perturbations(static_cast<int>(vd*ds->patternSize())) {
  IncRef(ds);
  IncRef(random);
  perturbed_positions = new int[ds->patternSize()];
}

template<typename T>
SaltPepperNoiseDataSet<T>::~SaltPepperNoiseDataSet() {
  DecRef(ds);
  DecRef(random);
  delete[] perturbed_positions;
}

template<typename T>
int SaltPepperNoiseDataSet<T>::getPattern(int index, T *pat) {
  int ret = ds->getPattern(index, pat);
  int sz  = ds->patternSize();
  random->shuffle(sz, perturbed_positions);
  for (int i=0; i<number_of_perturbations; ++i) 
    pat[perturbed_positions[i]] = (random->rand() < 0.5) ? zero : one;
  return ret;
}

template<typename T>
int SaltPepperNoiseDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for SaltPepperNoiseDataSet!!!\n");
  return 0;  
}

//-------------------------------------------------------------

template <typename T>
StepDataSet<T>::StepDataSet(DataSet<T> *ds, int step):
  DataSet<T>(),
  ds(ds), step(step){
  IncRef(ds);
}

template<typename T>
StepDataSet<T>::~StepDataSet() {
  DecRef(ds);
}
template<typename T> int StepDataSet<T>::getPattern(int index, T *pat) {

  // The original position in the dataset
  int dst_index = (index * this->step);
  
  int ret = this->ds->getPattern(dst_index,pat);
  return ret;
}

template<typename T> int StepDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1, "Method putPattern forbidden for StepDataset!!\n");
  return 0;
}
//-------------------------------------------------------------
template <typename T>
DerivDataSet<T>::DerivDataSet(DataSet<T> *ds,
			      bool deriv0, bool deriv1, bool deriv2) :
  DataSet<T>(),
  ds(ds), deriv0(deriv0), deriv1(deriv1), deriv2(deriv2) {
  IncRef(ds);
  origpatternsz = ds->patternSize();
  patternsz     = 0;
  if (deriv0) patternsz++;
  if (deriv1) patternsz++;
  if (deriv2) patternsz++;
  patternsz *= origpatternsz;
  numpatterns = ds->numPatterns();
  left1  = new T[origpatternsz];
  left2  = new T[origpatternsz];
  right1 = new T[origpatternsz];
  right2 = new T[origpatternsz];
  orig   = (!deriv0 && deriv2) ? (new T[origpatternsz]) : 0;
}

template<typename T>
DerivDataSet<T>::~DerivDataSet() {
  DecRef(ds);
  delete[] left1;
  delete[] left2;
  delete[] right1;
  delete[] right2;
  if (!deriv0 && deriv2) delete[] orig;
}

template<typename T>
int DerivDataSet<T>::getPattern(int index, T *pat) {

  ds->getPattern(clamp(index-2,0,numpatterns-1),left2);
  ds->getPattern(clamp(index-1,0,numpatterns-1),left1);
  ds->getPattern(clamp(index+1,0,numpatterns-1),right1);
  ds->getPattern(clamp(index+2,0,numpatterns-1),right2);

  T *aux_pat = pat;
  if (deriv0) {
    ds->getPattern(index, aux_pat);
    aux_pat += origpatternsz;
  }
  if (deriv1) {
    for (int i=0; i<origpatternsz; ++i)
      aux_pat[i] = (-right2[i] + 8*right1[i] -8*left1[i] + left2[i])/12.0;
    aux_pat += origpatternsz;
  }
  if (deriv2) {
    T *aux_orig = pat;
    if (!deriv0) {
      aux_orig = orig;
      ds->getPattern(index, orig);
    }
    for (int i=0; i<origpatternsz; ++i)
      aux_pat[i] = (-right2[i] + 16*right1[i] -30*aux_orig[i] + 16*left1[i] -left2[i])/12.0;
  }
  return patternsz;
}

template<typename T>
int DerivDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for DerivDataSet!!!\n");
  return 0;  
}

//-------------------------------------------------------------


template<typename T>
CacheDataSet<T>::CacheDataSet(DataSet<T> *ds, int **word2cache,
			      int *word2cache_sizes,
			      T *decays,
			      int voc_size,
			      int cache_size,
			      T near_zero,
			      int begin_token_id,
			      int end_token_id,
			      int null_token_id,
			      int cache_stop_token_id) :
  DataSet<T>(),
  ds(ds),word2cache(word2cache),
  word2cache_sizes(word2cache_sizes), decays(decays),
  voc_size(voc_size),
  cache_size(cache_size),
  near_zero(near_zero),
  begin_token_id(begin_token_id),
  end_token_id(end_token_id),
  null_token_id(null_token_id),
  cache_stop_token_id(cache_stop_token_id)
{
  IncRef(ds);
  float max_decay = 0.0f;
  for (int i=1; i<=cache_size; ++i)
    if (max_decay < decays[i]) max_decay = decays[i];
  max_history = log(near_zero)/log(max_decay);
  if (ds->patternSize() > 1) {
    ERROR_EXIT(1,"Incorrect dataset size!!!\n");
  }
}

template<typename T>
CacheDataSet<T>::~CacheDataSet() {
  DecRef(ds);
  for (int i=0; i<voc_size; ++i)
    delete[] word2cache[i];
  delete[] word2cache;
  delete[] word2cache_sizes;
  delete[] decays;
}

template<typename T>
int CacheDataSet<T>::getPattern(int index, T *pat) {
  T output;
  // ponemos la cache a cero
  for (int i=0; i<cache_size; ++i) pat[i] = 0.0f;
  // buscamos el fin de la frase anterior, la frase actual no forma
  // parte de la cache
  int j;
  for (j=index-1; j >= 0; --j) {
    ds->getPattern(j, &output);
    int wid = int(output);
    if (wid == end_token_id) break;
  }
  int k=0;
  // para toda la historia de la cache...
  for (j=j-1; j >= 0 && k < max_history; --j) {
    ds->getPattern(j, &output);
    int wid = int(output);
    // en caso de ser un stop, paramos
    if (wid == cache_stop_token_id) break;
    // solo se tiene en cuenta si la palabra es diferente a: bos, eos, null
    if (wid != begin_token_id && wid != end_token_id && wid != null_token_id) {
      // vector de indices
      int *idx_v = word2cache[wid];
      if (idx_v != 0) {
	// para cada indice...
	for (int i=0; i<word2cache_sizes[wid]; ++i) {
	  int idx = word2cache[wid][i];
	  // calculamos el decay que toca a esta posicion
	  if (pat[idx-1] < near_zero) {
	    float exp_decay = powf(decays[idx], k);
	    // si el indice es < near_zero, entonces le ponemos el
	    // valor del decay
	    if (exp_decay > near_zero) pat[idx-1] = exp_decay;
	    // si es < near_zero le ponemos un 0.0 exacto
	    else pat[idx-1] = 0.0f;
	  }
	}
      }
      // procesamos una palabra mas
      ++k;
    }
  }
  return cache_size;
}

template<typename T>
int CacheDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for CacheDataSet!!!\n");
  return 0;
}

//-------------------------------------------------------------

template <typename T>
SubAndDivNormalizationDataSet<T>::
SubAndDivNormalizationDataSet(DataSet<T> *ds, T *sub, T *div) :
  DataSet<T>(),
  ds(ds) {
  IncRef(ds);
  const int psize=ds->patternSize();
  this->sub = new T[psize];
  memcpy(this->sub, sub, sizeof(T)*psize);
  this->div = new T[psize];
  for (int i=0; i<psize; ++i) this->div[i] = 1.0/div[i];
}

template<typename T>
SubAndDivNormalizationDataSet<T>::~SubAndDivNormalizationDataSet() {
  DecRef(ds);
  delete[] sub;
  delete[] div;
}

template<typename T>
int SubAndDivNormalizationDataSet<T>::getPattern(int index, T *pat) {
  const int ret = ds->getPattern(index, pat);
  const int sz  = ds->patternSize();
  april_assert(ret==sz);
  for (int i=0; i<sz; ++i)
    pat[i] = (pat[i]-sub[i])*div[i];
  return ret;
}

template<typename T>
int SubAndDivNormalizationDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for SubAndDivNormalizationDataSet!!!\n");
  return 0;  
}

//-------------------------------------------------------------

template <typename T>
ClampDataSet<T>::
ClampDataSet(DataSet<T> *ds, float lower, float upper) :
  DataSet<T>(),
  ds(ds), lower(lower), upper(upper) {
  IncRef(ds);
}

template<typename T>
ClampDataSet<T>::~ClampDataSet() {
  DecRef(ds);
}

template<typename T>
int ClampDataSet<T>::getPattern(int index, T *pat) {
  const int ret = ds->getPattern(index, pat);
  const int sz  = ds->patternSize();
  april_assert(ret==sz);
  for (int i=0; i<sz; ++i)
    pat[i] = clamp(pat[i], lower, upper);
  return ret;
}

template<typename T>
int ClampDataSet<T>::putPattern(int index, const T *pat) {
  UNUSED_VARIABLE(index);
  UNUSED_VARIABLE(pat);
  ERROR_EXIT(1,"Method putPattern forbidden for ClampDataSet!!!\n");
  return 0;  
}

//----------------------------------------------------------

#endif // DATASET_CC_H
