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
#ifndef UTILDATASETTOKEN_H
#define UTILDATASETTOKEN_H

#include "unused_variable.h"
#include "token_base.h"
#include "token_matrix.h"
#include "token_vector.h"
#include "table_of_token_codes.h"
#include "dataset.h"
#include "datasetFloat.h"
#include "wrapper.h"
#include "vector.h"

class DataSetToken : public Referenced {
public:
  virtual ~DataSetToken() { }
  /// Number of patterns in the set
  virtual int numPatterns()=0;
  /// Size of each pattern.
  virtual int patternSize()=0;
  /// Get the pattern index to the vector pat
  virtual Token *getPattern(int index)=0;
  /// Get the pattern index to the vector pat
  virtual Token *getPatternBunch(const int *indexes,unsigned int bunch_size) {
    Token *output;
    Token *aux_token = getPattern(indexes[0]);
    TokenCode token_code = aux_token->getTokenCode();
    switch(token_code) {
    case table_of_token_codes::token_matrix: {
      int dims[2]   = { static_cast<int>(bunch_size),
			patternSize() };
      MatrixFloat *output_mat = new MatrixFloat(2, dims, CblasColMajor);
      TokenMatrixFloat *output_mat_token  = new TokenMatrixFloat(output_mat);
      output           = output_mat_token;
      unsigned int i   = 0;
      MatrixFloat::sliding_window window(output_mat, 0, 0, 0, 0, 0);
      while(!window.isEnd()) {
	IncRef(aux_token);
	TokenMatrixFloat *aux_mat_token;
	aux_mat_token = aux_token->convertTo<TokenMatrixFloat*>();
	
	MatrixFloat *submat = window.getMatrix();
	submat->copy(aux_mat_token->getMatrix());
	delete submat;
	
	DecRef(aux_token);
	if ( (++i) < bunch_size) aux_token = getPattern(indexes[i]);
	window.next();
      }
      break;
    }
    default: {
      TokenBunchVector *output_token_vector = new TokenBunchVector(bunch_size);
      output         = output_token_vector;
      unsigned int i = 0;
      do {
	(*output_token_vector)[i] = aux_token;
	IncRef(aux_token);
	if ( (++i) < bunch_size) aux_token = getPattern(indexes[i]);
	else break;
      } while(true);
      break;
    }
    }
    return output;
  }
  /// Put the given vector pat at pattern index
  virtual void putPattern(int index, Token *pat)=0;
  /// Put the pattern bunch
  virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
			       Token *pat)=0;
};

class DataSetTokenVector : public DataSetToken {
  april_utils::vector<Token*> data;
  int pattern_size;
public:
  DataSetTokenVector(int pattern_size) : pattern_size(pattern_size) { }
  virtual ~DataSetTokenVector() {
    for (unsigned int i=0; i<data.size(); ++i) DecRef(data[i]);
  }
  void push_back(Token *token) {
    data.push_back(token);
    IncRef(token);
  }
  /// Number of patterns in the set
  virtual int numPatterns() { return static_cast<int>(data.size()); }
  /// Size of each pattern.
  virtual int patternSize() { return pattern_size; }
  /// Get the pattern index to the vector pat
  virtual Token *getPattern(int index) {
    if (index < 0 || index >= numPatterns())
      return 0;
    return data[index]->clone();
  }
  /// Put the given vector pat at pattern index
  virtual void putPattern(int index, Token *pat) {
    if (index < 0 || index >= numPatterns()) return;
    AssignRef(data[index], pat);
  }
  /// Put the pattern bunch
  virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
			       Token *pat) {
    UNUSED_VARIABLE(indexes);
    UNUSED_VARIABLE(bunch_size);
    UNUSED_VARIABLE(pat);
    ERROR_EXIT(128, "Not implemented!!!\n");    
  }
};

class UnionDataSetToken : public DataSetToken {
  april_utils::vector<DataSetToken*> ds_array;
  april_utils::vector<int>           sum_patterns;
  int pattern_size;

public:
  UnionDataSetToken(DataSetToken **ds_array, unsigned int size) :
    pattern_size(0) {
    sum_patterns.push_back(0);
    for (unsigned int i=0; i<size; ++i) push_back(ds_array[i]);
  }
  UnionDataSetToken() : pattern_size(0) {
    sum_patterns.push_back(0);
  }
  virtual ~UnionDataSetToken() {
    for (unsigned int i=0; i<ds_array.size(); ++i) DecRef(ds_array[i]);
  }
  void push_back(DataSetToken *ds) {
    ds_array.push_back(ds);
    IncRef(ds);
    if (pattern_size == 0)
      pattern_size = ds->patternSize();
    else if (pattern_size != ds->patternSize())
      ERROR_EXIT2(128, "Incorrect pattern size, expected %d, found %d\n",
		  pattern_size, ds->patternSize());
    sum_patterns.push_back(sum_patterns.back() + ds->numPatterns());
  }
  /// Number of patterns in the set
  virtual int numPatterns() { return sum_patterns.back(); }
  /// Size of each pattern.
  virtual int patternSize() { return pattern_size; }
  /// Get the pattern index to the vector pat
  virtual Token *getPattern(int index) {
    if (index < 0 || index >= numPatterns())
      return 0;
    int izq,der,m;
    izq = 0; der = static_cast<int>(sum_patterns.size());
    do {
      m = (izq+der)/2;
      if (sum_patterns[m] <= index) 
	izq = m; 
      else 
	der = m;
    } while (izq < der-1);
    return ds_array[izq]->getPattern(index-sum_patterns[izq]);
  }
  /// Put the given vector pat at pattern index
  virtual void putPattern(int index, Token *pat) {
    UNUSED_VARIABLE(index);
    UNUSED_VARIABLE(pat);
    ERROR_EXIT(128, "Not implemented!!!\n");
  }
  /// Put the pattern bunch
  virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
			       Token *pat) {
    UNUSED_VARIABLE(indexes);
    UNUSED_VARIABLE(bunch_size);
    UNUSED_VARIABLE(pat);
    ERROR_EXIT(128, "Not implemented!!!\n");
  }
};

class DataSetFloat2TokenWrapper : public DataSetToken {
  MatrixFloat  *aux_mat;
  DataSetFloat *ds;
public:
  DataSetFloat2TokenWrapper(DataSetFloat *ds) : ds(ds) {
    IncRef(ds);
    int dims[2] = { 1, ds->patternSize() };
    aux_mat = new MatrixFloat(2, dims, CblasColMajor);
  }
  virtual ~DataSetFloat2TokenWrapper() {
    DecRef(ds);
    delete aux_mat;
  }
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize(); }
  Token *getPattern(int index) {
    int dims[2] = { 1, patternSize() };
    MatrixFloat *mat = new MatrixFloat(2, dims, CblasColMajor);
    TokenMatrixFloat *token = new TokenMatrixFloat(mat);
    FloatGPUMirroredMemoryBlock *mem_block = mat->getRawDataAccess();
    float *mem_ptr = mem_block->getPPALForWrite();
    ds->getPattern(index, mem_ptr);
    return token;
  }
  Token *getPatternBunch(const int *indexes, unsigned int bunch_size) {
    int dims[2] = { static_cast<int>(bunch_size), patternSize() };
    MatrixFloat *mat = new MatrixFloat(2, dims, CblasColMajor);
    TokenMatrixFloat *token = new TokenMatrixFloat(mat);
    FloatGPUMirroredMemoryBlock *aux_mem_block = aux_mat->getRawDataAccess();
    float *aux_mem = aux_mem_block->getPPALForWrite();
    int pattern_size = patternSize();
    int num_patterns = numPatterns();
    MatrixFloat::sliding_window window(mat, 0, 0, 0, 0, 0);
    for (unsigned int i=0; i<bunch_size; ++i) {
      april_assert(!window.isEnd());
      april_assert(0 <= indexes[i] && indexes[i] < num_patterns);
      ds->getPattern(indexes[i], aux_mem);
      MatrixFloat *submat = window.getMatrix();
      submat->copy(aux_mat);
      delete submat;
      window.next();
    }
    return token;
  }
  void putPattern(int index, Token *pat) {
    if (pat->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT(128, "Incorrect token type, expected token matrix\n");
    TokenMatrixFloat *token_matrix = pat->convertTo<TokenMatrixFloat*>();
    MatrixFloat *mat = token_matrix->getMatrix();
    FloatGPUMirroredMemoryBlock *mem_block = mat->getRawDataAccess();
    const float *v = mem_block->getPPALForRead();
    ds->putPattern(index, v);
  }
  
  void putPatternBunch(const int *indexes,unsigned int bunch_size,
		       Token *pat) {
    if (pat->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT(128, "Incorrect token type, expected token matrix\n");
    TokenMatrixFloat *token_matrix = pat->convertTo<TokenMatrixFloat*>();
    MatrixFloat *mat = token_matrix->getMatrix();
    if (mat->getNumDim() != 2)
      ERROR_EXIT(128, "Only allowed for 2-dim matrices\n");
    FloatGPUMirroredMemoryBlock *aux_mem_block = aux_mat->getRawDataAccess();
    float *aux_mem = aux_mem_block->getPPALForWrite();
    int pattern_size = patternSize();
    for (unsigned int i=0; i<bunch_size; ++i) {
      int coords[2] = { static_cast<int>(i), 0 };
      int sizes[2]  = { 1, pattern_size };
      MatrixFloat *submat = new MatrixFloat(mat, coords, sizes, false);
      aux_mat->copy(submat);
      delete submat;
      ds->putPattern(indexes[i], aux_mem);
    }
  }
};

#endif // UTILDATASETFLOAT_H
