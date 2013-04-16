/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include "token_memory_block.h"

TokenMemoryBlock::TokenMemoryBlock() : mem_block(0), used_size(0) { }

TokenMemoryBlock::TokenMemoryBlock(unsigned int size) : used_size(0) {
  mem_block = new GPUMirroredMemoryBlock(size);
}

TokenMemoryBlock::~TokenMemoryBlock() {
  delete mem_block;
}

void TokenMemoryBlock::resize(unsigned int size) {
  if (size > mem_block->getSize()) {
    delete mem_block;
    mem_block = new GPUMirroredMemoryBlock(size);
  }
  used_size = size;
}

Token *TokenMemoryBlock::clone() const {
  TokenMemoryBlock *token = new TokenMemoryBlock(mem_block->getSize());
  token->used_size = used_size;
  doScopy(mem_block->getSize(),
	  mem_block, 0, 1,
	  token->mem_block, 0, 1,
	  GlobalConf::use_cuda);
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

static Token *TokenMemoryBlock::fromString(constString &cs) {
  // NOT IMPLEMENTED
  return 0;
}
