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
#include "dataset.h"
#include "datasetFloat.h"
#include "wrapper.h"

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
    ds->getPattern(pattern_size, mem_ptr);
    return token;
  }
  Token *getPatternBunch(const int *indexes, unsigned int bunch_size) {
    TokenMemoryBlock *token = new TokenMemoryBlock(bunch_size*pattern_size);
    FloatGPUMirroredMemoryBlock *mem_block = token->getMemBlock();
    unsigned int mem_ptr_shift = 0;
    for (unsigned int i=0; i<bunch_size; ++i, ++mem_ptr_shift) {
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
      doScopy(pattern_size,
	      mem_block, mem_ptr_shift, bunch_size,
	      aux_mem_block, 0, 1,
	      false);
      ds->putPattern(indexes[i], aux_mem_block->getPPALForRead());
    }
  }
};

#endif // UTILDATASETFLOAT_H
