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
#include "bind_april_io.h"
#include "bind_matrix.h"
#include "bind_tokens.h"
#include "token_matrix.h"

using namespace AprilMath::MatrixExt::Reductions;
using namespace Basics;
//BIND_END

//BIND_HEADER_H
#include "loss_function.h"
#include "mse_loss_function.h"
#include "mae_loss_function.h"
#include "cross_entropy_loss_function.h"
#include "multiclass_cross_entropy_loss_function.h"
#include "batch_fmeasure_micro_avg_loss_function.h"
#include "batch_fmeasure_macro_avg_loss_function.h"
#include "zero_one_loss_function.h"

using namespace ANN;
//BIND_END

/////////////////////////////////////////////////////
//                   LossFunction                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME Serializable aprilio.serializable

//BIND_LUACLASSNAME LossFunction ann.loss
//BIND_CPP_CLASS    LossFunction
//BIND_SUBCLASS_OF  LossFunction Serializable

//BIND_CONSTRUCTOR LossFunction
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD LossFunction compute_loss
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, AuxToken);
  LUABIND_CHECK_PARAMETER(2, AuxToken);
  AprilUtils::SharedPtr<Token> input, target;
  LUABIND_GET_PARAMETER(1, AuxToken, input);
  LUABIND_GET_PARAMETER(2, AuxToken, target);
  MatrixFloat *loss = obj->computeLoss(input.get(), target.get());
  if (loss) {
    LUABIND_RETURN(float, matSum(loss)/loss->getDimSize(0));
    LUABIND_RETURN(MatrixFloat, loss);
  }
  else {
    LUABIND_RETURN_NIL();
  }
}
//BIND_END

//BIND_METHOD LossFunction accum_loss
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_CHECK_ARGN(<=,2);
  float tr_loss = 0.0f;
  MatrixFloat *loss;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, MatrixFloat);
    LUABIND_GET_PARAMETER(1, MatrixFloat, loss);
  }
  else {
    LUABIND_CHECK_PARAMETER(1, float);
    LUABIND_CHECK_PARAMETER(2, MatrixFloat);
    LUABIND_GET_PARAMETER(1, float, tr_loss);
    LUABIND_GET_PARAMETER(2, MatrixFloat, loss);
  }
  obj->accumLoss(loss);
  if (argn == 1)
    LUABIND_RETURN(MatrixFloat, loss);
  else {
    LUABIND_RETURN(float, tr_loss);
    LUABIND_RETURN(MatrixFloat, loss);
  }
}
//BIND_END

//BIND_METHOD LossFunction gradient
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1, AuxToken);
  LUABIND_CHECK_PARAMETER(2, AuxToken);
  AprilUtils::SharedPtr<Token> input, target;
  LUABIND_GET_PARAMETER(1, AuxToken, input);
  LUABIND_GET_PARAMETER(2, AuxToken, target);
  AprilUtils::SharedPtr<Token> error( obj->computeGradient(input.get(),
                                                           target.get()) );
  LUABIND_RETURN(AuxToken, error);
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
  unsigned int size;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
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
  unsigned int size;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
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
  unsigned int size;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
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

//BIND_LUACLASSNAME NonPairedCrossEntropyLossFunction ann.loss.non_paired_cross_entropy
//BIND_CPP_CLASS    NonPairedCrossEntropyLossFunction
//BIND_SUBCLASS_OF  NonPairedCrossEntropyLossFunction LossFunction

//BIND_CONSTRUCTOR NonPairedCrossEntropyLossFunction
{
  unsigned int size;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
  obj=new NonPairedCrossEntropyLossFunction(size);
  LUABIND_RETURN(NonPairedCrossEntropyLossFunction, obj);
}
//BIND_END

//BIND_METHOD NonPairedCrossEntropyLossFunction clone
{
  LUABIND_RETURN(NonPairedCrossEntropyLossFunction,
		 dynamic_cast<NonPairedCrossEntropyLossFunction*>(obj->clone()));
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
  unsigned int size;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
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

//BIND_LUACLASSNAME NonPairedMultiClassCrossEntropyLossFunction ann.loss.non_paired_multi_class_cross_entropy
//BIND_CPP_CLASS    NonPairedMultiClassCrossEntropyLossFunction
//BIND_SUBCLASS_OF  NonPairedMultiClassCrossEntropyLossFunction LossFunction

//BIND_CONSTRUCTOR NonPairedMultiClassCrossEntropyLossFunction
{
  unsigned int size;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
  obj=new NonPairedMultiClassCrossEntropyLossFunction(size);
  LUABIND_RETURN(NonPairedMultiClassCrossEntropyLossFunction, obj);
}
//BIND_END

//BIND_METHOD NonPairedMultiClassCrossEntropyLossFunction clone
{
  LUABIND_RETURN(NonPairedMultiClassCrossEntropyLossFunction,
		 dynamic_cast<NonPairedMultiClassCrossEntropyLossFunction*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//                BATCH FMEASURE                   //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME BatchFMeasureMicroAvgLossFunction ann.loss.batch_fmeasure_micro_avg
//BIND_CPP_CLASS    BatchFMeasureMicroAvgLossFunction
//BIND_SUBCLASS_OF  BatchFMeasureMicroAvgLossFunction LossFunction

//BIND_CONSTRUCTOR BatchFMeasureMicroAvgLossFunction
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  unsigned int size=0;
  float beta=1.0f;
  bool complement=false;
  if (argn > 0) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "size", "beta", "complement", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, complement, bool, complement, false);
  }
  obj=new BatchFMeasureMicroAvgLossFunction(size, beta, complement);
  LUABIND_RETURN(BatchFMeasureMicroAvgLossFunction, obj);
}
//BIND_END

//BIND_METHOD BatchFMeasureMicroAvgLossFunction clone
{
  LUABIND_RETURN(BatchFMeasureMicroAvgLossFunction,
		 dynamic_cast<BatchFMeasureMicroAvgLossFunction*>(obj->clone()));
}
//BIND_END

///////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BatchFMeasureMacroAvgLossFunction ann.loss.batch_fmeasure_macro_avg
//BIND_CPP_CLASS    BatchFMeasureMacroAvgLossFunction
//BIND_SUBCLASS_OF  BatchFMeasureMacroAvgLossFunction LossFunction

//BIND_CONSTRUCTOR BatchFMeasureMacroAvgLossFunction
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  unsigned int size=0;
  float beta=1.0f;
  bool complement=false;
  if (argn > 0) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "size", "beta", "complement", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, complement, bool, complement, false);
  }
  obj=new BatchFMeasureMacroAvgLossFunction(size, beta, complement);
  LUABIND_RETURN(BatchFMeasureMacroAvgLossFunction, obj);
}
//BIND_END

//BIND_METHOD BatchFMeasureMacroAvgLossFunction clone
{
  LUABIND_RETURN(BatchFMeasureMacroAvgLossFunction,
		 dynamic_cast<BatchFMeasureMacroAvgLossFunction*>(obj->clone()));
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
  unsigned int size;
  float TH;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, size, 0);
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

