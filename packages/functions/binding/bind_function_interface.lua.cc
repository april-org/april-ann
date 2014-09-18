/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include "bind_tokens.h"
//BIND_END

//BIND_HEADER_H
#include "function_interface.h"
#include "identity_function.h"
using namespace Functions;
using Basics::Token;
//BIND_END

//BIND_LUACLASSNAME FunctionInterface functions
//BIND_CPP_CLASS    FunctionInterface

//BIND_CONSTRUCTOR FunctionInterface
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD FunctionInterface get_input_size
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(uint, obj->getInputSize());
}
//BIND_END

//BIND_METHOD FunctionInterface get_output_size
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(uint, obj->getOutputSize());
}
//BIND_END

//BIND_METHOD FunctionInterface calculate
{
  LUABIND_CHECK_ARGN(==,1);
  Token *input;
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_RETURN(Token, obj->calculate(input));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME IdentityFunction functions
//BIND_CPP_CLASS IdentityFunction
//BIND_SUBCLASS_OF IdentityFunction FunctionInterface

//BIND_CONSTRUCTOR IdentityFunction
{
  LUABIND_CHECK_ARGN(==,0);
  obj = new IdentityFunction();
  LUABIND_RETURN(IdentityFunction, obj);
}
//BIND_END
