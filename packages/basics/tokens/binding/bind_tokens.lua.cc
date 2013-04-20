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

//BIND_LUACLASSNAME Token token
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
