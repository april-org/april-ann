/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include "bind_dataset.h"
//BIND_END

//BIND_HEADER_H
#include "function_interface.h"
#include "datasetfloat_producer.h"
#include "datasetfloat_consumer.h"

using namespace Functions;

//BIND_END

//BIND_LUACLASSNAME FunctionBase __function_base__
//BIND_CPP_CLASS    FunctionBase

//BIND_CONSTRUCTOR FunctionBase
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD FunctionBase get_input_size
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(uint, obj->getInputSize());
}
//BIND_END

//BIND_METHOD FunctionBase get_output_size
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(uint, obj->getOutputSize());
}
//BIND_END

///////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FloatFloatFunctionInterface __function_interface_float_float__
//BIND_CPP_CLASS    FloatFloatFunctionInterface

//BIND_SUBCLASS_OF FloatFloatFunctionInterface FunctionBase

//BIND_CONSTRUCTOR FloatFloatFunctionInterface
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END


//BIND_METHOD FloatFloatFunctionInterface calculate
//DOC_BEGIN
// calculate(input) : table

/// Execute the calculate method of a function, and returns a Lua table with the function output

/// @param input A Lua table with the functions input
//DOC_END
{
  unsigned int input_size, output_size = obj->getOutputSize();
  float *input, *output = new float[output_size];
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_TABLE_GETN(1, input_size);
  if (input_size != obj->getInputSize())
    LUABIND_ERROR("Incorrect input size!!!\n");
  input = new float[input_size];
  LUABIND_TABLE_TO_VECTOR(1, float, input, input_size);
  if (!obj->calculate(input, input_size, output, output_size))
    LUABIND_ERROR("Impossible to execute calculate method!!!\n");
  LUABIND_VECTOR_TO_NEW_TABLE(float, output, output_size);
  LUABIND_RETURN_FROM_STACK(-1);
  delete[] input;
  delete[] output;
}
//BIND_END

//BIND_METHOD FloatFloatFunctionInterface calculate_in_pipeline
//DOC_BEGIN
// calculate_in_pipeline{ consumer, producer }

/// Execute the calculate in pipeline method of a function

/// @param input A Lua table with the functions input
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "producer",
		     "consumer",
		     "input_size",
		     "output_size",
		     0);
  // de tipo float => float
  FloatDataProducer *producer;
  unsigned int       input_size;
  FloatDataConsumer *consumer;
  unsigned int	     output_size;
  
  LUABIND_GET_TABLE_PARAMETER(1, producer, FloatDataProducer, producer);
  LUABIND_GET_TABLE_PARAMETER(1, consumer, FloatDataConsumer, consumer);
  LUABIND_GET_TABLE_PARAMETER(1, input_size, uint, input_size);
  LUABIND_GET_TABLE_PARAMETER(1, output_size, uint, output_size);

  obj->calculateInPipeline(producer, input_size,
			   consumer, output_size);
}
//BIND_END

////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FloatDataProducer __float_data_producer__
//BIND_CPP_CLASS    FloatDataProducer

//BIND_CONSTRUCTOR  FloatDataProducer
{
  LUABIND_ERROR("this is an abstract class");
  return 0;
}
//BIND_END

//BIND_DESTRUCTOR   FloatDataProducer
{
}
//BIND_END

//BIND_METHOD FloatDataProducer reset
{
  obj->reset();
}
//BIND_END


//BIND_LUACLASSNAME FloatDataConsumer __float_data_consumer__
//BIND_CPP_CLASS    FloatDataConsumer

//BIND_CONSTRUCTOR  FloatDataConsumer
{
  LUABIND_ERROR("this is an abstract class");
  return 0;
}
//BIND_END

//BIND_DESTRUCTOR   FloatDataConsumer
{
}
//BIND_END

//BIND_METHOD FloatDataConsumer reset
{
  obj->reset();
}
//BIND_END

//BIND_LUACLASSNAME LogFloatDataProducer __logfloat_data_producer__
//BIND_CPP_CLASS    LogFloatDataProducer

//BIND_CONSTRUCTOR  LogFloatDataProducer
{
  LUABIND_ERROR("this is an abstract class");
  return 0;
}
//BIND_END

//BIND_DESTRUCTOR   LogFloatDataProducer
{
}
//BIND_END

//BIND_METHOD LogFloatDataProducer reset
{
  obj->reset();
}
//BIND_END



//BIND_LUACLASSNAME DoubleDataProducer __double_data_producer__
//BIND_CPP_CLASS    DoubleDataProducer

//BIND_CONSTRUCTOR  DoubleDataProducer
{
  LUABIND_ERROR("this is an abstract class");
  return 0;
}
//BIND_END

//BIND_DESTRUCTOR   DoubleDataProducer
{
}
//BIND_END

//BIND_METHOD DoubleDataProducer reset
{
  obj->reset();
}
//BIND_END



//BIND_LUACLASSNAME LogFloatDataConsumer __logfloat_data_consumer__
//BIND_CPP_CLASS    LogFloatDataConsumer

//BIND_CONSTRUCTOR  LogFloatDataConsumer
{
  LUABIND_ERROR("this is an abstract class");
  return 0;
}
//BIND_END

//BIND_DESTRUCTOR   LogFloatDataConsumer
{
}
//BIND_END

//BIND_METHOD LogFloatDataConsumer reset
{
  obj->reset();
}
//BIND_END

////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME DataSetFloatProducer functions.producer.dataset
//BIND_CPP_CLASS    DataSetFloatProducer

//BIND_SUBCLASS_OF  DataSetFloatProducer FloatDataProducer

//BIND_CONSTRUCTOR  DataSetFloatProducer
{
  DataSetFloat *ds;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  obj = new DataSetFloatProducer(ds);
  LUABIND_RETURN(DataSetFloatProducer, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME DataSetFloatConsumer functions.consumer.dataset
//BIND_CPP_CLASS    DataSetFloatConsumer

//BIND_SUBCLASS_OF  DataSetFloatConsumer FloatDataConsumer

//BIND_CONSTRUCTOR  DataSetFloatConsumer
{
  DataSetFloat *ds;
  LUABIND_GET_PARAMETER(1, DataSetFloat, ds);
  obj = new DataSetFloatConsumer(ds);
  LUABIND_RETURN(DataSetFloatConsumer, obj);
}
//BIND_END
