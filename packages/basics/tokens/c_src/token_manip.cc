/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013 Francisco Zamora-Martinez
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
#include "token_manip.h"
#include "table_of_token_codes.h"
#include "token_base.h"
#include "token_vector.h"
#include "token_memory_block.h"
#include "error_print.h"
#include "wrapper.h"

/// This function pushes in a bunch of patterns a new pattern. Patterns are
/// of type TokenMemoryBlock.
void pushTokenMemBlockAt(unsigned int bunch_pos, Token *bunch, Token *pat) {
  if (bunch->getTokenCode() != table_of_token_codes::token_memory_block ||
      pat->getTokenCode() != table_of_token_codes::token_memory_block)
    ERROR_EXIT(82, "Incorrect token code found, expected token_memory_block\n");
  TokenMemoryBlock *bunch_mem_block_token, *pat_mem_block_token;
  bunch_mem_block_token = bunch->converTo<TokenMemoryBlock*>();
  pat_mem_block_token   = pat->converTo<TokenMemoryBlock*>();
  //
  if (bunch_mem_block_token->getUsedSize() % pat_mem_block_token->getUsedSize() != 0)
    ERROR_EXIT(128, "Incorrect sizes");
  unsigned int bunch_size = bunch_mem_block_token->getUsedSize()/pat->getUsedSize();
  unsigned int size = pat_mem_block_token->getUsedSize();
  // copy in row major order, depending on bunch size
  doScopy(size,
	  bunch_mem_block_token->getMemBlock(), bunch_pos, bunch_size,
	  pat_mem_block_token->getMemBlock(), 0, 1,
	  false);
}

/// This function puhses in a bunch of patterns a new pattern. Patterns could
/// be any combination of TokenBunchVector's which internally contains
/// a TokenMemoryBlock.
void pushTokenAt(unsigned int bunch_pos, Token *&bunch, Token *pat) {
  if (bunch == 0) bunch = pat->clone();
  if (bunch->getTokenCode() != pat->getTokenCode())
    ERROR_EXIT(80, "Token types must be equal, found different.\n");
  switch(bunch->getTokenCode()) {
  case table_of_token_codes::vector_Tokens:
    {
      TokenBunchVector *bunch_vector = bunch->converTo<TokenBunchVector*>();
      TokenBunchVector *pat_vector   = bunch->converTo<TokenBunchVector*>();
      if (bunch_vector->size() != pat_vector->size())
	ERROR_EXIT(81, "Found different token sizes, expected equal\n");
      for (unsigned int i=0; i<bunch_vector->size(); ++i)
	pushTokenAt(bunch_pos, bunch_vector[i], pat_vector[i]);
    }
    break;
  case table_of_token_codes::token_memory_block:
    pushTokenMemBlockAt(bunch_pos, bunch, pat);
    break;
  default:
    ERROR_EXIT(82, "Incorrect token code found, "
	       "expected token_memory_block or vector_Tokens\n");
  }
}
