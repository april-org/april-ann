/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#include "bind_matrix.h"
//BIND_END

//BIND_HEADER_H
#include "loss_function.h"
#include "mse_loss_function.h"
#include "mae_loss_function.h"
#include "cross_entropy_loss_function.h"
#include "multiclass_cross_entropy_loss_function.h"
#include "local_fmeasure_loss_function.h"
#include "zero_one_loss_function.h"

using namespace ANN;

//BIND_END

/////////////////////////////////////////////////////
//                   LossFunction                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LossFunction ann.loss.__base__
//BIND_CPP_CLASS    LossFunction

//BIND_CONSTRUCTOR LossFunction
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD LossFunction compute_loss
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, Token);
  LUABIND_CHECK_PARAMETER(2, Token);
  Token *input, *target;
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_GET_PARAMETER(2, Token, target);
  MatrixFloat *loss = obj->computeLoss(input, target);
  LUABIND_RETURN(MatrixFloat, loss);
}
//BIND_END

//BIND_METHOD LossFunction accum_loss
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  MatrixFloat *loss;
  LUABIND_GET_PARAMETER(1, MatrixFloat, loss);
  obj->accumLoss(loss);
  LUABIND_RETURN(MatrixFloat, loss);
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
  Token *error = obj->computeGradient(input, target);
  LUABIND_RETURN(Token, error);
}
//BIND_END

//BIND_METHOD LossFunction get_accum_loss
{
  float loss = obj->getAccumLoss();
  float variance = obj->getAccumLossVariance();
  LUABIND_RETURN(float, loss);
  LUABIND_RETURN(float, variance);
}
//BIND_END

//BIND_METHOD LossFunction reset
{
  obj->reset();
}
//BIND_END

//BIND_METHOD LossFunction clone
{
  LUABIND_RETURN(LossFunction, obj->clone());
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

//BIND_METHOD MSELossFunction clone
{
  LUABIND_RETURN(MSELossFunction, dynamic_cast<MSELossFunction*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//                       MAE                       //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME MAELossFunction ann.loss.mae
//BIND_CPP_CLASS    MAELossFunction
//BIND_SUBCLASS_OF  MAELossFunction LossFunction

//BIND_CONSTRUCTOR MAELossFunction
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, number);
  unsigned int size;
  LUABIND_GET_PARAMETER(1, uint, size);
  obj=new MAELossFunction(size);
  LUABIND_RETURN(MAELossFunction, obj);
}
//BIND_END

//BIND_METHOD MAELossFunction clone
{
  LUABIND_RETURN(MAELossFunction, dynamic_cast<MAELossFunction*>(obj->clone()));
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

//BIND_METHOD CrossEntropyLossFunction clone
{
  LUABIND_RETURN(CrossEntropyLossFunction,
		 dynamic_cast<CrossEntropyLossFunction*>(obj->clone()));
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

//BIND_METHOD MultiClassCrossEntropyLossFunction clone
{
  LUABIND_RETURN(MultiClassCrossEntropyLossFunction,
		 dynamic_cast<MultiClassCrossEntropyLossFunction*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//                LOCAL FMEASURE                   //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LocalFMeasureLossFunction ann.loss.local_fmeasure
//BIND_CPP_CLASS    LocalFMeasureLossFunction
//BIND_SUBCLASS_OF  LocalFMeasureLossFunction LossFunction

//BIND_CONSTRUCTOR LocalFMeasureLossFunction
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "size", "beta", "complement", (const char *)0);
  unsigned int size;
  float beta;
  bool complement;
  LUABIND_GET_TABLE_PARAMETER(1, size, uint, size);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, complement, bool, complement, false);
  obj=new LocalFMeasureLossFunction(size, beta, complement);
  LUABIND_RETURN(LocalFMeasureLossFunction, obj);
}
//BIND_END

//BIND_METHOD LocalFMeasureLossFunction clone
{
  LUABIND_RETURN(LocalFMeasureLossFunction,
		 dynamic_cast<LocalFMeasureLossFunction*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//                ZERO-ONE LOSS                    //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ZeroOneLossFunction ann.loss.zero_one
//BIND_CPP_CLASS    ZeroOneLossFunction
//BIND_SUBCLASS_OF  ZeroOneLossFunction LossFunction

//BIND_CONSTRUCTOR ZeroOneLossFunction
{
  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_CHECK_ARGN(<=,2);
  unsigned int size;
  float TH;
  LUABIND_GET_PARAMETER(1, uint, size);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, TH, 0.5f);
  obj=new ZeroOneLossFunction(size, TH);
  LUABIND_RETURN(ZeroOneLossFunction, obj);
}
//BIND_END

//BIND_METHOD ZeroOneLossFunction clone
{
  LUABIND_RETURN(ZeroOneLossFunction,
		 dynamic_cast<ZeroOneLossFunction*>(obj->clone()));
}
//BIND_END

