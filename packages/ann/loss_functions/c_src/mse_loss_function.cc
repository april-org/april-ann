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
#include "mse_loss_function.h"

namespace ANN {

  MSELMSELossFunction::MSELossFunction(unsigned int size) :
  Referenced(), LossFunction(size), accumulated_loss(0.0f) {
  }
  
  MSELMSELossFunction::~MSELossFunction() {
  }
  
  float MSELossFunction::addLoss(Token *_input, Token *target) {
    if (_input->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect input token type, expected memory block\n");
    if (target->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect target token type, expected memory block\n");
    //
    if (input != 0) DecRef(input);
    input = _input->convertTo<TokenMemoryBlock*>();
    IncRef(input);
    TokenMemoryBlock *input_mem_token = input->convertTo<TokenMemoryBlock*>();
    TokenMemoryBlock *target_mem_block = target->converTo<TokenMemoryBlock*>();
    if (input_mem_token->getUsedSize() != target_mem_block->getUsedSize())
      ERROR_EXIT(128, "Different token sizes found\n");
    //
    unsigned int bunch_size = input->getUsedSize() / size;
    float loss = 0.5f * doMSELossFunction(input, target, 0.0f, size, bunch_size,
					  input_mem_token->getCudaFlag());
    accumulated_loss += loss;
    return loss;
  }

  Token *MSELossFunction::computeGrandient(Token *_input, Token *target) {
    if (_input->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect token type, expected memory block\n");
    if (target->getTokenCode() != table_of_token_codes::token_mem_block)
      ERROR_EXIT(128, "Incorrect target token type, expected memory block\n");
    //
    if (input != _input) {
      if (input != 0) DecRef(input);
      input = _input->convertTo<TokenMemoryBlock*>();
      IncRef(input);
    }
    TokenMemoryBlock *target_mem_block = target->converTo<TokenMemoryBlock*>();
    if (input_mem_token->getUsedSize() != target_mem_block->getUsedSize())
      ERROR_EXIT(128, "Different token sizes found\n");
    //
    unsigned int bunch_size = input->getUsedSize() / size;
    error_output->resize(bunch_size);
    doAccumulateMSEGradient(input, target, error_output, 0.0f, size, bunch_size,
			    input_mem_token->getCudaFlag());
    return error_output;
  }
  
  float MSELossFunction::getTotalLoss() {
    return accumulated_loss;
  }
   
  void MSELossFunction::reset() {
    accumulated_loss = 0.0f;
    doVectorSetToZero(error_output->getMemBlock(),
		      error_output->getMaxSize(),
		      1, 0, use_cuda);
  }
}
