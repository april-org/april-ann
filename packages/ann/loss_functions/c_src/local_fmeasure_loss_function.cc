/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios, Francisco Zamora-Martinez
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
#include "token_memory_block.h"
#include "local_fmeasure_loss_function.h"
#include "wrapper.h"

namespace ANN {

  LocalFMeasureLossFunction::LocalFMeasureLossFunction(unsigned int size,
						       float beta,
						       bool complement_output) :
    LossFunction(size), beta(beta), complement_output(complement_output),
    accumulated_loss(0.0f), N(0) {
  }
  
  LocalFMeasureLossFunction::~LocalFMeasureLossFunction() {
  }
  
  float LocalFMeasureLossFunction::addLoss(Token *input, Token *target) {
    if (input->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect input token type, expected memory block\n");
    if (target->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect target token type, expected memory block\n");
    //
    TokenMemoryBlock *input_mem_token = input->convertTo<TokenMemoryBlock*>();
    TokenMemoryBlock *target_mem_block = target->convertTo<TokenMemoryBlock*>();
    if (input_mem_token->getUsedSize() != target_mem_block->getUsedSize())
      ERROR_EXIT(128, "Different token sizes found\n");
    //
    unsigned int bunch_size = input_mem_token->getUsedSize() / size;
    Gab = 0.0f;
    Hab = 0.0f;
    float loss = doLocalFMeasureLossFunction(input_mem_token->getMemBlock(),
					     target_mem_block->getMemBlock(),
					     size, bunch_size,
					     beta, Gab, Hab,
					     complement_output,
					     input_mem_token->getCudaFlag());
    accumulated_loss += loss;
    ++N;
    return loss;
  }
  
  Token *LocalFMeasureLossFunction::computeGradient(Token *input, Token *target) {
    if (input->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type, expected memory block\n");
    if (target->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect target token type, expected memory block\n");
    //
    TokenMemoryBlock *input_mem_token  = input->convertTo<TokenMemoryBlock*>();
    TokenMemoryBlock *target_mem_block = target->convertTo<TokenMemoryBlock*>();
    if (input_mem_token->getUsedSize() != target_mem_block->getUsedSize())
      ERROR_EXIT2(128, "Different token sizes found, input=%d  target=%d\n",
		  input_mem_token->getUsedSize(),
		  target_mem_block->getUsedSize());
    //
    unsigned int bunch_size = input_mem_token->getUsedSize() / size;
    TokenMemoryBlock *error_mem_block;
    error_mem_block = new TokenMemoryBlock(input_mem_token->getUsedSize());
    AssignRef(error_output, error_mem_block);
    doComputeLocalFMeasureGradient(target_mem_block->getMemBlock(),
				   error_mem_block->getMemBlock(),
				   size, bunch_size,
				   beta, Gab, Hab, complement_output,
				   input_mem_token->getCudaFlag());
    return error_output;
  }
  
  float LocalFMeasureLossFunction::getAccumLoss() {
    return accumulated_loss/N;
  }
  
  void LocalFMeasureLossFunction::reset() {
    LossFunction::reset();
    accumulated_loss = 0.0f;
    N = 0;
  }
}
