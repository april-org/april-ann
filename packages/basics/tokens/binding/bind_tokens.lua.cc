/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
//BIND_HEADER_C

//BIND_END

//BIND_HEADER_H
#include "token_base.h"
#include "token_memory_block.h"
#include "token_vector.h"
//BIND_END

//BIND_LUACLASSNAME Token tokens.base
//BIND_CPP_CLASS    Token

//BIND_CONSTRUCTOR Token
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD Token clone
{
  LUABIND_RETURN(Token, obj->clone());
}
//BIND_END

//BIND_METHOD Token convert_to_memblock
{
  TokenMemoryBlock *token_memblock = obj->convertTo<TokenMemoryBlock*>();
  LUABIND_RETURN(TokenMemoryBlock, token_memblock);
}
//BIND_END

////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TokenMemoryBlock tokens.memblock
//BIND_CPP_CLASS    TokenMemoryBlock
//BIND_SUBCLASS_OF  TokenMemoryBlock Token

//BIND_CONSTRUCTOR TokenMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  unsigned int sz;
  LUABIND_TABLE_GETN(1, sz);
  float *vector = new float[sz];
  LUABIND_TABLE_TO_VECTOR(1, float, vector, sz);
  obj = new TokenMemoryBlock(sz);
  obj->setData(vector, sz);
  delete[] vector;
  LUABIND_RETURN(TokenMemoryBlock, obj);
}
//BIND_END

//BIND_METHOD TokenMemoryBlock to_table
{
  int sz = static_cast<int>(obj->getUsedSize());;
  const float *vector = obj->getMemBlock()->getPPALForRead();
  LUABIND_VECTOR_TO_NEW_TABLE(float, vector, sz);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END
