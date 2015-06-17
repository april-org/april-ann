/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
//BIND_HEADER_H
#include "token_base.h"
#include "token_memory_block.h"
#include "token_matrix.h"
#include "token_null.h"
#include "token_sparse_matrix.h"
#include "token_vector.h"
#include "smart_ptr.h"

using namespace Basics;

bool lua_isAuxToken(lua_State *L, int n);
AprilUtils::SharedPtr<Token> lua_toAuxToken(lua_State *L, int n);
void lua_pushAuxToken(lua_State *L, AprilUtils::SharedPtr<Token> &value);
//BIND_END

//BIND_HEADER_C
#include "bind_matrix.h"
#include "bind_sparse_matrix.h"
#include "matrixFloat.h"
#include "sparse_matrixFloat.h"

namespace AprilUtils {
  template<> AprilUtils::SharedPtr<Basics::Token> LuaTable::
  convertTo< AprilUtils::SharedPtr<Basics::Token> >(lua_State *L, int idx) {
    return lua_toAuxToken(L, idx);
  }
  
  template<> void LuaTable::
  pushInto< AprilUtils::SharedPtr<Basics::Token> >(lua_State *L,
                                                   AprilUtils::SharedPtr<Basics::Token> ptr) {
    lua_pushAuxToken(L, ptr);
  }

  template<> bool LuaTable::
  checkType< AprilUtils::SharedPtr<Basics::Token> >(lua_State *L, int idx) {
    return lua_isAuxToken(L, idx);
  }
}

namespace Basics {
  int token_bunch_vector_iterator_function(lua_State *L) {
    // se llama con: local var_1, ... , var_n = _f(_s, _var) donde _s es
    // el estado invariante (en este caso el dataset) y _var es var_1 de
    // iteracion anterior (en este caso el indice)
    TokenBunchVector *obj = lua_toTokenBunchVector(L, 1);
    unsigned int index = static_cast<unsigned int>(lua_tonumber(L, 2)) + 1; // le sumamos uno
    if (index > obj->size()) {
      lua_pushnil(L); return 1;
    }
    lua_pushnumber(L, index);
    AprilUtils::SharedPtr<Token> token( (*obj)[index-1] );
    lua_pushAuxToken(L, token);
    return 2;
  }

} // namespace Basics

bool lua_isAuxToken(lua_State *L, int n) {
  return lua_isSparseMatrixFloat(L,n) || lua_isMatrixFloat(L,n) || lua_isToken(L,n);
}

AprilUtils::SharedPtr<Token> lua_toAuxToken(lua_State *L, int n) {
  if (lua_isMatrixFloat(L, n)) {
    MatrixFloat *mat = lua_toMatrixFloat(L,n);
    return new TokenMatrixFloat(mat);
  }
  else if (lua_isSparseMatrixFloat(L, n)) {
    SparseMatrixFloat *mat = lua_toSparseMatrixFloat(L,n);
    return new TokenSparseMatrixFloat(mat);
  }
  else if (lua_isnil(L, n)) {
    return TokenNull::getInstance();
  }
  return lua_toToken(L,n);
}

void lua_pushAuxToken(lua_State *L, AprilUtils::SharedPtr<Token> &value) {
  if (value.empty()) {
    lua_pushTokenNull(L, TokenNull::getInstance());
  }
  else {
    switch(value->getTokenCode()) {
    case Basics::table_of_token_codes::token_matrix:
      lua_pushMatrixFloat(L, ((TokenMatrixFloat*)value.get())->getMatrix());
      break;
    case Basics::table_of_token_codes::token_sparse_matrix:
      lua_pushSparseMatrixFloat(L, ((TokenSparseMatrixFloat*)value.get())->getMatrix());
      break;
    case Basics::table_of_token_codes::vector_Tokens:
      lua_pushTokenBunchVector(L, (TokenBunchVector*)value.get());
      break;
    case Basics::table_of_token_codes::token_null:
      lua_pushTokenNull(L, TokenNull::getInstance());
      break;
    default:
      lua_pushToken(L, value.get());
    }
  }
}

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

//BIND_METHOD Token get_matrix
{
  switch(obj->getTokenCode()) {
  case table_of_token_codes::token_matrix: {
    TokenMatrixFloat *token_matrix = obj->convertTo<TokenMatrixFloat*>();
    LUABIND_RETURN(MatrixFloat, token_matrix->getMatrix());
    break;
  }
  case table_of_token_codes::token_sparse_matrix: {
    TokenSparseMatrixFloat *token_sparse_matrix;
    token_sparse_matrix = obj->convertTo<TokenSparseMatrixFloat*>();
    LUABIND_RETURN(SparseMatrixFloat, token_sparse_matrix->getMatrix());
    break;
  }
  default:
    ;
  }
}
//BIND_END

//BIND_METHOD Token convert_to_memblock
{
  if (obj->getTokenCode() == table_of_token_codes::token_mem_block) {
    TokenMemoryBlock *token_memblock = obj->convertTo<TokenMemoryBlock*>();
    LUABIND_RETURN(TokenMemoryBlock, token_memblock);
  }
}
//BIND_END

//BIND_METHOD Token convert_to_bunch_vector
{
  if (obj->getTokenCode() == table_of_token_codes::vector_Tokens) {
    TokenBunchVector *token_bunch_vector = obj->convertTo<TokenBunchVector*>();
    LUABIND_RETURN(TokenBunchVector, token_bunch_vector);
  }
}
//BIND_END

////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TokenNull tokens.null
//BIND_CPP_CLASS    TokenNull
//BIND_SUBCLASS_OF  TokenNull Token

//BIND_CONSTRUCTOR TokenNull
{
  LUABIND_RETURN(TokenNull, TokenNull::getInstance());
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
  int sz = static_cast<int>(obj->getUsedSize());
  const float *vector = obj->getMemBlock()->getPPALForRead();
  LUABIND_VECTOR_TO_NEW_TABLE(float, vector, sz);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TokenMatrixFloat tokens.matrix
//BIND_CPP_CLASS    TokenMatrixFloat
//BIND_SUBCLASS_OF  TokenMatrixFloat Token

//BIND_CONSTRUCTOR TokenMatrixFloat
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  obj = new TokenMatrixFloat(mat);
  LUABIND_RETURN(TokenMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD TokenMatrixFloat get_matrix
{
  LUABIND_RETURN(MatrixFloat, obj->getMatrix());
}
//BIND_END

////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TokenSparseMatrixFloat tokens.sparse_matrix
//BIND_CPP_CLASS    TokenSparseMatrixFloat
//BIND_SUBCLASS_OF  TokenSparseMatrixFloat Token

//BIND_CONSTRUCTOR TokenSparseMatrixFloat
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, SparseMatrixFloat);
  SparseMatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, SparseMatrixFloat, mat);
  obj = new TokenSparseMatrixFloat(mat);
  LUABIND_RETURN(TokenSparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD TokenSparseMatrixFloat get_matrix
{
  LUABIND_RETURN(SparseMatrixFloat, obj->getMatrix());
}
//BIND_END

////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TokenVectorGeneric tokens.vector
//BIND_CPP_CLASS    TokenVectorGeneric
//BIND_SUBCLASS_OF  TokenVectorGeneric Token

//BIND_CONSTRUCTOR TokenVectorGeneric
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD TokenVectorGeneric size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD TokenVectorGeneric clear
{
  obj->clear();
}
//BIND_END

////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TokenBunchVector tokens.vector.bunch
//BIND_CPP_CLASS    TokenBunchVector
//BIND_SUBCLASS_OF  TokenBunchVector TokenVectorGeneric

//BIND_CONSTRUCTOR TokenBunchVector
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  if (argn == 1) {
    if (lua_isnumber(L, 1)) {
      unsigned int size;
      LUABIND_CHECK_PARAMETER(1, uint);
      LUABIND_GET_PARAMETER(1, uint, size);
      obj = new TokenBunchVector(size);
    }
    else {
      int sz;
      LUABIND_TABLE_GETN(1, sz);
      obj = new TokenBunchVector(sz);
      for (unsigned int i=0; i<static_cast<unsigned int>(sz); ++i) {
        lua_pushnumber(L, i+1); // OJO: +1
        lua_gettable(L, 1);
        AprilUtils::SharedPtr<Token> aux(lua_toAuxToken(L,-1));
        lua_pop(L,1);
	(*obj)[i] = aux.get();
	IncRef(aux.get());
      }
    }
  }
  else obj = new TokenBunchVector();
  LUABIND_RETURN(TokenBunchVector, obj);
}
//BIND_END

//BIND_METHOD TokenBunchVector at
{
  LUABIND_CHECK_ARGN(==, 1);
  unsigned int pos;
  LUABIND_CHECK_PARAMETER(1, uint);
  LUABIND_GET_PARAMETER(1, uint, pos);
  AprilUtils::SharedPtr<Token> token( (*obj)[pos-1] );
  LUABIND_RETURN(AuxToken, token);
}
//BIND_END

//BIND_METHOD TokenBunchVector set
{
  LUABIND_CHECK_ARGN(==, 2);
  unsigned int pos;
  AprilUtils::SharedPtr<Token> token;
  LUABIND_CHECK_PARAMETER(1, uint);
  LUABIND_CHECK_PARAMETER(2, AuxToken);
  LUABIND_GET_PARAMETER(1, uint, pos);
  LUABIND_GET_PARAMETER(2, AuxToken, token);
  (*obj)[pos-1] = token.get();
  IncRef(token.get());
  LUABIND_RETURN(TokenBunchVector, obj);
}
//BIND_END

//BIND_METHOD TokenBunchVector push_back
{
  LUABIND_CHECK_ARGN(==, 1);
  AprilUtils::SharedPtr<Token> token;
  LUABIND_CHECK_PARAMETER(1, AuxToken);
  LUABIND_GET_PARAMETER(1, AuxToken, token);
  obj->TokenBunchVector::push_back(token.get());
  LUABIND_RETURN(TokenBunchVector, obj);
}
//BIND_END

//BIND_METHOD TokenBunchVector iterate
// para iterar con un for index,value in obj:iterate() do ... end
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,token_bunch_vector_iterator_function);
  LUABIND_RETURN(TokenBunchVector,obj);
  LUABIND_RETURN(int,0);
}
//BIND_END
