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
#ifndef TOKEN_MEMORY_BLOCK_H
#define TOKEN_MEMORY_BLOCK_H

#include "disallow_class_methods.h"
#include "gpu_mirrored_memory_block.h"
#include "token_base.h"
#include "unused_variable.h"

namespace Basics {

  class TokenMemoryBlock : public Token {
    APRIL_DISALLOW_COPY_AND_ASSIGN(TokenMemoryBlock);
    AprilMath::FloatGPUMirroredMemoryBlock *mem_block;
    unsigned int used_size;
  public:
    TokenMemoryBlock();
    TokenMemoryBlock(unsigned int size);
    ~TokenMemoryBlock();
    void setData(float *data, unsigned int size);
    AprilMath::FloatGPUMirroredMemoryBlock *getMemBlock() { return mem_block; }
    unsigned int getUsedSize() const { return used_size; }
    unsigned int getMaxSize() const { return mem_block?mem_block->getSize():0; }
    void resize(unsigned int size);
    Token *clone() const;
    AprilUtils::buffer_list* toString();
    AprilUtils::buffer_list* debugString(const char *prefix, int debugLevel);
    TokenCode getTokenCode() const;
    static Token *fromString(AprilUtils::constString &cs) {
      // NOT IMPLEMENTED
      UNUSED_VARIABLE(cs);
      return 0;
    }
    void clear() { used_size = 0; }
    bool getInCudag() { return mem_block->getInCuda(); }
    void printDebug() {
      const float *data = mem_block->getPPALForRead();
      for (unsigned int i=0; i<used_size; ++i)
        printf ("%f ", data[i]);
      printf("\n");
    }
    void setToZero(bool use_cuda);
  };

} // namespace Basics

#endif // TOKEN_MEMORY_BLOCK_H
