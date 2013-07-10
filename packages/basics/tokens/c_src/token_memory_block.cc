/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
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
#include "wrapper.h"
#include "gpu_mirrored_memory_block.h"
#include "token_memory_block.h"

TokenMemoryBlock::TokenMemoryBlock() : mem_block(0), used_size(0) { }

TokenMemoryBlock::TokenMemoryBlock(unsigned int size) :
  mem_block(0), used_size(0) {
  resize(size);
}

TokenMemoryBlock::~TokenMemoryBlock() {
  delete mem_block;
}

void TokenMemoryBlock::setData(float *data, unsigned int size) {
  resize(size);
  float *mem_ptr = mem_block->getPPALForWrite();
  for (unsigned int i=0; i<size; ++i)
    mem_ptr[i] = data[i];
}

void TokenMemoryBlock::resize(unsigned int size) {
  if (mem_block == 0)
    mem_block = new FloatGPUMirroredMemoryBlock(size);
  else if (size > mem_block->getSize()) {
    delete mem_block;
    mem_block = new FloatGPUMirroredMemoryBlock(size);
  }
  used_size = size;
}

Token *TokenMemoryBlock::clone() const {
  TokenMemoryBlock *token = new TokenMemoryBlock(mem_block->getSize());
  token->used_size = used_size;
  doScopy(mem_block->getSize(),
	  mem_block, 0, 1,
	  token->mem_block, 0, 1,
	  false);
  return token;
}

buffer_list* TokenMemoryBlock::toString() {
  // NOT IMPLEMENTED
  return 0;
}

buffer_list* TokenMemoryBlock::debugString(const char *prefix, int debugLevel) {
  // NOT IMPLEMENTED
  return 0;
}

TokenCode TokenMemoryBlock::getTokenCode() const {
  return table_of_token_codes::token_mem_block;
}

void TokenMemoryBlock::setToZero(bool use_cuda) {
  doFill(mem_block->getSize(), mem_block, 1, 0, use_cuda);
}
