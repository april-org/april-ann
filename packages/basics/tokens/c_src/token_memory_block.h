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
#ifndef TOKEN_MEMORY_BLOCK_H
#define TOKEN_MEMORY_BLOCK_H

#include "gpu_mirrored_memory_block.h"
#include "token_base.h"

class TokenMemoryBlock : public Token {
  FloatGPUMirroredMemoryBlock *mem_block;
  unsigned int used_size;
public:
  TokenMemoryBlock();
  TokenMemoryBlock(unsigned int size);
  ~TokenMemoryBlock();
  FloatGPUMirroredMemoryBlock *getMemBlock() { return mem_block; }
  unsigned int getUsedSize() const { return used_size; }
  unsigned int getMaxSize() const { return mem_block->getSize(); }
  void resize(unsigned int size);
  Token *clone() const;
  buffer_list* toString();
  buffer_list* debugString(const char *prefix, int debugLevel);
  TokenCode getTokenCode() const;
  static Token *fromString(constString &cs) {
    // NOT IMPLEMENTED
    return 0;
  }
  void clear() { used_size = 0; }
};

#endif // TOKEN_MEMORY_BLOCK_H
