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
#include "token_memory_block.h"
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
  virtual Token *getPatternBunch(const int *indexes,unsigned int bunch_size)=0;
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
    return data[index];
  }
  /// Get the pattern index to the vector pat
  virtual Token *getPatternBunch(const int *indexes,unsigned int bunch_size) {
    Token *output;
    Token *aux_token = getPattern(indexes[0]);
    TokenCode token_code = aux_token->getTokenCode();
    switch(token_code) {
    case table_of_token_codes::token_mem_block: {
      TokenMemoryBlock *output_mem_token;
      output_mem_token = new TokenMemoryBlock(bunch_size * pattern_size);
      output           = output_mem_token;
      unsigned int i   = 0;
      unsigned int pos = 0;
      do {
	IncRef(aux_token);
	TokenMemoryBlock *aux_mem_block_token;
	aux_mem_block_token = aux_token->convertTo<TokenMemoryBlock*>();
	doScopy(pattern_size,
		aux_mem_block_token->getMemBlock(), 0, 1,
		output_mem_token->getMemBlock(), pos++, bunch_size,
		aux_mem_block_token->getCudaFlag());
	DecRef(aux_token);
	if ( (++i) < bunch_size) aux_token = getPattern(indexes[i]);
	else break;
      } while(true);
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
  virtual void putPattern(int index, Token *pat) {
    if (index < 0 || index >= numPatterns()) return;
    AssignRef(data[index], pat);
  }
  /// Put the pattern bunch
  virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
			       Token *pat) {
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
  /// Get the pattern index to the vector pat
  virtual Token *getPatternBunch(const int *indexes,unsigned int bunch_size) {
    Token *output;
    Token *aux_token = getPattern(indexes[0]);
    TokenCode token_code = aux_token->getTokenCode();
    switch(token_code) {
    case table_of_token_codes::token_mem_block: {
      TokenMemoryBlock *output_mem_token;
      output_mem_token = new TokenMemoryBlock(bunch_size * pattern_size);
      output           = output_mem_token;
      unsigned int i   = 0;
      unsigned int pos = 0;
      do {
	IncRef(aux_token);
	TokenMemoryBlock *aux_mem_block_token;
	aux_mem_block_token = aux_token->convertTo<TokenMemoryBlock*>();
	doScopy(pattern_size,
		aux_mem_block_token->getMemBlock(), 0, 1,
		output_mem_token->getMemBlock(), pos++, bunch_size,
		aux_mem_block_token->getCudaFlag());
	DecRef(aux_token);
	if ( (++i) < bunch_size) aux_token = getPattern(indexes[i]);
	else break;
      } while(true);
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
  virtual void putPattern(int index, Token *pat) {
    ERROR_EXIT(128, "Not implemented!!!\n");
  }
  /// Put the pattern bunch
  virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
			       Token *pat) {
    ERROR_EXIT(128, "Not implemented!!!\n");
  }
};

class DataSetFloat2TokenWrapper : public DataSetToken {
  FloatGPUMirroredMemoryBlock *aux_mem_block;
  DataSetFloat *ds;
  int           pattern_size;
  int           num_patterns;
public:
  DataSetFloat2TokenWrapper(DataSetFloat *ds) : ds(ds) {
    IncRef(ds);
    pattern_size  = ds->patternSize();
    num_patterns  = ds->numPatterns();
    aux_mem_block = new FloatGPUMirroredMemoryBlock(pattern_size);
  }
  virtual ~DataSetFloat2TokenWrapper() {
    DecRef(ds);
    delete aux_mem_block;
  }
  int numPatterns() { return num_patterns; }
  int patternSize() { return pattern_size; }
  Token *getPattern(int index) {
    TokenMemoryBlock *token = new TokenMemoryBlock(pattern_size);
    FloatGPUMirroredMemoryBlock *mem_block = token->getMemBlock();
    float *mem_ptr = mem_block->getPPALForWrite();
    ds->getPattern(index, mem_ptr);
    return token;
  }
  Token *getPatternBunch(const int *indexes, unsigned int bunch_size) {
    TokenMemoryBlock *token = new TokenMemoryBlock(bunch_size*pattern_size);
    FloatGPUMirroredMemoryBlock *mem_block = token->getMemBlock();
    unsigned int mem_ptr_shift = 0;
    for (unsigned int i=0; i<bunch_size; ++i, ++mem_ptr_shift) {
      assert(0 <= indexes[i] && indexes[i] < num_patterns);
      ds->getPattern(indexes[i],
		     aux_mem_block->getPPALForWrite());
      doScopy(pattern_size, aux_mem_block, 0, 1,
	      mem_block, mem_ptr_shift, bunch_size,
	      false);
    }
    return token;
  }
  void putPattern(int index, Token *pat) {
    if (pat->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type, expected token memory block\n");
    TokenMemoryBlock *token_mem_block = pat->convertTo<TokenMemoryBlock*>();
    FloatGPUMirroredMemoryBlock *mem_block = token_mem_block->getMemBlock();
    const float *v = mem_block->getPPALForRead();
    ds->putPattern(index, v);
  }
  
  void putPatternBunch(const int *indexes,unsigned int bunch_size,
		       Token *pat) {
    if (pat->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type, expected token memory block\n");
    TokenMemoryBlock *token_mem_block = pat->convertTo<TokenMemoryBlock*>();
    FloatGPUMirroredMemoryBlock *mem_block = token_mem_block->getMemBlock();
    unsigned int mem_ptr_shift = 0;
    for (unsigned int i=0; i<bunch_size; ++i, ++mem_ptr_shift) {
      assert(0 <= indexes[i] && indexes[i] < num_patterns);
      doScopy(pattern_size,
	      mem_block, mem_ptr_shift, bunch_size,
	      aux_mem_block, 0, 1,
	      false);
      ds->putPattern(indexes[i], aux_mem_block->getPPALForRead());
    }
  }
};

#endif // UTILDATASETFLOAT_H
