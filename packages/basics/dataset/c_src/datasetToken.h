/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef UTILDATASETTOKEN_H
#define UTILDATASETTOKEN_H

#include "token_base.h"
#include "dataset.h"
#include "datasetFloat.h"

class DataSetToken : public Referenced {
  virtual ~DataSet() { }
  /// Number of patterns in the set
  virtual int numPatterns()=0;
  /// Size of each pattern.
  virtual int patternSize()=0;
  /// Get the pattern index to the vector pat
  virtual Token *getPattern(int index)=0;
  /// Put the given vector pat at pattern index
  virtual int putPattern(int index, const Token *pat)=0;
}

class DataSetFloat2TokenWrapper : public DataSetToken {
  float        *aux;
  DataSetFloat *ds;
 public:
  DataSet2TokenWrapper(DataSet<T> *ds) : ds(ds) {
    aux = new float[ds->patternSize];    
    IncRef(ds);
  }
  virtual ~DataSet2TokenWrapper() {
    DecRef(ds);
    delete[] aux;
  }
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize();   }
  Token *getPattern(int index) {
    unsigned int psize = static_cast<unsigned int>(ds->getPattern(index, aux));
    return new TokenVectorFloat(psize, aux);
  }
  int putPattern(int index, const Token *pat) {
    const float *v = pat->data();
    ds->putPattern(index, v);
  }
};


#endif // UTILDATASETFLOAT_H
