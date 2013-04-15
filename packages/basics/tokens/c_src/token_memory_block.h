#ifndef TOKEN_MEMORY_BLOCK_H
#define TOKEN_MEMORY_BLOCK_H

#include "gpu_mirrored_memory_block.h"
#include "token_base.h"

class TokenMemoryBlock : public TokenBase {
  GPUMirroredMemoryBlock *mem_block;
  unsigned int used_size;
public:
  TokenMemoryBlock();
  TokenMemoryBlock(unsigned int size);
  ~TokenMemoryBlock();
  const GPUMirroredMemoryBlock *getMemBlock() const { return mem_block; }
  GPUMirroredMemoryBlock *getMemBlock() { return mem_block; }
  void getUsedSize() const { return used_size; }
  void getMaxSize() const { return mem_block->getSize(); }
  void resize(unsigned int size);
  Token *clone() const;
  buffer_list* toString();
  buffer_list* debugString(const char *prefix, int debugLevel);
  TokenCode getTokenCode() const;
  static Token *fromString(constString &cs);
};

#endif // TOKEN_VECTOR_H
