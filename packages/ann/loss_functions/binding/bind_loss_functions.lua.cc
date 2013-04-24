/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#include "loss_function.h"
#include "mse_loss_function.h"
#include "cross_entropy_loss_function.h"
#include "multiclass_cross_entropy_loss_function.h"

using namespace ANN;

//BIND_END

/////////////////////////////////////////////////////
//                   LossFunction                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LossFunction ann.loss
//BIND_CPP_CLASS    LossFunction

//BIND_CONSTRUCTOR LossFunction
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD LossFunction loss
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, Token);
  LUABIND_CHECK_PARAMETER(2, Token);
  Token *input, *target;
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_GET_PARAMETER(2, Token, target);
  float loss = obj->addLoss(input, target);
  LUABIND_RETURN(float, loss);
}
//BIND_END

//BIND_METHOD LossFunction gradient
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, Token);
  LUABIND_CHECK_PARAMETER(2, Token);
  Token *input, *target;
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_GET_PARAMETER(2, Token, target);
  Token *error = obj->computeGrandient(input, target);
  LUABIND_RETURN(Token, error);
}
//BIND_END

//BIND_METHOD LossFunction accum_loss
{
  float loss = obj->getAccumLoss();
  LUABIND_RETURN(float, loss);
}
//BIND_END

//BIND_METHOD LossFunction reset
{
  obj->reset();
}
//BIND_END

/////////////////////////////////////////////////////
//                       MSE                       //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME MSELossFunction ann.loss.mse
//BIND_CPP_CLASS    MSELossFunction
//BIND_SUBCLASS_OF  MSELossFunction LossFunction

//BIND_CONSTRUCTOR MSELossFunction
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  unsigned int size;
  LUABIND_GET_PARAMETER(1, uint, size);
  obj=new MSELossFunction(size);
  LUABIND_RETURN(MSELossFunction, obj);
}
//BIND_END

/////////////////////////////////////////////////////
//                  CROSS ENTROPY                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME CrossEntropyLossFunction ann.loss.cross_entropy
//BIND_CPP_CLASS    CrossEntropyLossFunction
//BIND_SUBCLASS_OF  CrossEntropyLossFunction LossFunction

//BIND_CONSTRUCTOR CrossEntropyLossFunction
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  unsigned int size;
  LUABIND_GET_PARAMETER(1, uint, size);
  obj=new CrossEntropyLossFunction(size);
  LUABIND_RETURN(CrossEntropyLossFunction, obj);
}
//BIND_END

/////////////////////////////////////////////////////
//          MULTI-CLASS CROSS ENTROPY              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME MultiClassCrossEntropyLossFunction ann.loss.multi_class_cross_entropy
//BIND_CPP_CLASS    MultiClassCrossEntropyLossFunction
//BIND_SUBCLASS_OF  MultiClassCrossEntropyLossFunction LossFunction

//BIND_CONSTRUCTOR MultiClassCrossEntropyLossFunction
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  unsigned int size;
  LUABIND_GET_PARAMETER(1, uint, size);
  obj=new MultiClassCrossEntropyLossFunction(size);
  LUABIND_RETURN(MultiClassCrossEntropyLossFunction, obj);
}
//BIND_END
