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
#include "bind_mtrand.h"
#include "bind_function_interface.h"
//BIND_END

//BIND_HEADER_H
#include "datasetFloat.h"
#include "matrixFloat.h"
#include "trainsuper.h"

using namespace Trainable;
using namespace Functions;

//BIND_END

//BIND_LUACLASSNAME FloatFloatFunctionInterface __function_interface_float_float__
//BIND_LUACLASSNAME TrainableSupervised __trainable__
//BIND_CPP_CLASS    TrainableSupervised
//BIND_SUBCLASS_OF  TrainableSupervised FloatFloatFunctionInterface

//BIND_CONSTRUCTOR TrainableSupervised
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD TrainableSupervised set_option
//DOC_BEGIN
// set_option(name_str, value)

/// Set the value of a learning parameter.

/// @param name_str A lua string with the name of the option.
/// @param value A lua number with the value for this option.
//DOC_END
{
  LUABIND_CHECK_ARGN(==,2);
  
  const char *name_str;
  double value;
  
  LUABIND_GET_PARAMETER(1, string, name_str);
  LUABIND_GET_PARAMETER(2, double, value);
  
  obj->setOption(name_str, value);
}
//BIND_END

//BIND_METHOD TrainableSupervised has_option
//DOC_BEGIN
// has_option(name_str)

/// Returns true/false depending on the existence of name_str as an
/// object option.

/// @param name_str A lua string with the name of the option.
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  
  const char *name_str;
  
  LUABIND_GET_PARAMETER(1, string, name_str);
  
  LUABIND_RETURN(bool, obj->hasOption(name_str));
}
//BIND_END


//BIND_METHOD TrainableSupervised get_option
//DOC_BEGIN
// double get_option(name_str)

/// Get the value of learning parameter.

/// @param name_str A lua string with the name of the option.
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  const char *name_str;
  LUABIND_GET_PARAMETER(1, string, name_str);
  LUABIND_RETURN(double, obj->getOption(name_str));
}
//BIND_END

//BIND_METHOD TrainableSupervised train_pattern
//DOC_BEGIN
// float train_pattern(input_table, target_output_table)

/// This functions train the mlp object one time with one pattern. The
/// sizes of the input_table and output_table must be corrects for the
/// corresponding mlp object.

/// @param input_table The input pattern. Is a lua table.
/// @param target_output_table The desired output pattern. Is a lua table.

//DOC_END

{
  LUABIND_CHECK_ARGN(==,2);
  unsigned int input_size, output_size;
  float *input, *output;
  
  LUABIND_TABLE_GETN(1, input_size);
  if (input_size != obj->getInputSize())
    LUABIND_FERROR2("Incorrect input size, was %d, expected %d\n",
		    input_size, obj->getInputSize());
  
  LUABIND_TABLE_GETN(2, output_size);
  if (output_size != obj->getOutputSize())
    LUABIND_FERROR2("Incorrect output size, was %d, expected %d\n",
		    output_size, obj->getOutputSize());
  
  input  = new float[input_size];
  output = new float[output_size];
  
  LUABIND_TABLE_TO_VECTOR(1, float, input,  input_size);
  LUABIND_TABLE_TO_VECTOR(2, float, output, output_size);
  
  float error = obj->trainOnePattern(input, output);
  
  LUABIND_RETURN(float, error);
}
//BIND_END

//BIND_METHOD TrainableSupervised validate_pattern
{
  LUABIND_CHECK_ARGN(==,2);
  unsigned int input_size, output_size;
  float *input, *output;
  
  LUABIND_TABLE_GETN(1, input_size);
  if (input_size != obj->getInputSize())
    LUABIND_FERROR2("Incorrect input size, was %d, expected %d\n",
		    input_size, obj->getInputSize());
  
  LUABIND_TABLE_GETN(2, output_size);
  if (output_size != obj->getOutputSize())
    LUABIND_FERROR2("Incorrect output size, was %d, expected %d\n",
		    output_size, obj->getOutputSize());
  
  input  = new float[input_size];
  output = new float[output_size];
  
  LUABIND_TABLE_TO_VECTOR(1, float, input,  input_size);
  LUABIND_TABLE_TO_VECTOR(2, float, output, output_size);
  
  float error = obj->validateOnePattern(input, output);
  
  LUABIND_RETURN(float, error);
}
//BIND_END

//BIND_METHOD TrainableSupervised train_dataset
//DOC_BEGIN
// float train_dataset(table params = { dataset input_dataset, dataset output_dataset, random shuffle = 0, replacement = input_dataset.numPatterns(), distribution = 0 })

/// Train the mlp with the parameters indicated in the table, and
/// returns the MSE of the dataset.

/// @param params.input_dataset A dataset object with each input pattern of the training.
/// @param params.output_dataset A dataset object with the corresponding output pattern for each input in input_dataset param.

//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "input_dataset",
		     "output_dataset",
		     "shuffle",
		     "replacement",
		     "distribution",
		     0);
  
  float mse = 0;
  
  // 2 tipos de funcionamiento:
  // a) pasando información a priori sobre las distribuciones
  // b) pasando 2 datasets: entrada y salida
  
  MTRand *mtrand;
  // si queremos shuffle obtenemos un mtrand
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, shuffle, MTRand, mtrand, 0);
  
  // si queremos estrategia con reemplazo, le pasamos un numero
  // positivo que corresponde al número de presentaciones de muestras
  // a la red:
  int replacement;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, replacement, int, replacement, 0);
  lua_getfield(L, 1, "distribution");
  if (lua_istable(L,-1)) {
    // utilizamos información sobre distribuciones a priori
    if (!mtrand) LUABIND_ERROR("mlp train with distribution requires shuffle");
    
    int num_classes;
    LUABIND_TABLE_GETN(-1, num_classes);
    DataSetFloat **v_ids = new DataSetFloat*[num_classes];
    DataSetFloat **v_ods = new DataSetFloat*[num_classes];
    double *aprioris = new double[num_classes];
    for (int i=0; i<num_classes; i++) {
      // tomamos una tripleta
      lua_rawgeti(L, -1, i+1);
      check_table_fields(L, -2, "input_dataset", "output_dataset", "probability", 0);
      // comprobamos que tiene input, output, etc.
      LUABIND_GET_TABLE_PARAMETER(-2, input_dataset,
				  DataSetFloat, v_ids[i]);
      LUABIND_GET_TABLE_PARAMETER(-2, output_dataset,
				  DataSetFloat, v_ods[i]);
      LUABIND_GET_TABLE_PARAMETER(-2, probability,
				  double, aprioris[i]);
      lua_pop(L,1); // nos cargamos la tabla i-ésima
    }
    mse = obj->trainDatasetWithDistribution(num_classes,
					    v_ids,
					    v_ods,
					    aprioris,
					    mtrand,
					    replacement);
    // liberamos memoria:
    delete[] v_ids;
    delete[] v_ods;
    delete[] aprioris;
    
  } else { // utilizamos datasets de entrada y de salida
    lua_pop(L,1); // seguramente es nil
    
    DataSetFloat *inputds,*outputds;
    
    LUABIND_GET_TABLE_PARAMETER(1, input_dataset,
				DataSetFloat, inputds);
    LUABIND_GET_TABLE_PARAMETER(1, output_dataset,
				DataSetFloat, outputds);
    
    if (mtrand != 0 && replacement != 0) { // shuffle with replacement
      mse = obj->trainDatasetWithReplacement(inputds,
					     outputds,
					     mtrand,
					     replacement);
    }
    else { // without replacement
      mse = obj->trainDataset(inputds, outputds, mtrand);
    }
  } // end of using input and output dataset
  LUABIND_RETURN(float, mse);
}
//BIND_END


//BIND_METHOD TrainableSupervised validate_dataset
//DOC_BEGIN
// float validate_dataset(table params = { dataset input_dataset, dataset output_dataset, random shuffle = 0, replacement = input_dataset.numPatterns(), distribution = 0 })

/// Validate a dataset with the mlp and the parameters indicated in the table, and
/// returns the MSE of the dataset.

/// @param params.input_dataset A dataset object with each input pattern of the training.
/// @param params.output_dataset A dataset object with the corresponding output pattern for each input in input_dataset param.

//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "input_dataset",
		     "output_dataset",
		     "shuffle",
		     "replacement",
		     0);
  
  float mse = 0;
  
  MTRand *mtrand;
  // si queremos shuffle obtenemos un mtrand
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, shuffle, MTRand, mtrand, 0);
  
  // si queremos estrategia con reemplazo, le pasamos un numero
  // positivo que corresponde al número de presentaciones de muestras
  // a la red:
  int replacement;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, replacement, int, replacement, 0);
  
  if (mtrand != 0 && replacement == 0)
    LUABIND_ERROR("shuffle is forbidden without replacement!!!");
  if (mtrand == 0 && replacement != 0)
    LUABIND_ERROR("shuffle is mandatory with replacement!!!");
  
  DataSetFloat *inputds,*outputds;
  
  LUABIND_GET_TABLE_PARAMETER(1, input_dataset,
			      DataSetFloat, inputds);
  LUABIND_GET_TABLE_PARAMETER(1, output_dataset,
			      DataSetFloat, outputds);
    
  if (replacement != 0) { // shuffle with replacement
    mse = obj->validateDatasetWithReplacement(inputds,
					      outputds,
					      mtrand,
					      replacement);
  }
  else { // without replacement
    mse = obj->validateDataset(inputds, outputds);
  } // end of using input and output dataset
  LUABIND_RETURN(float, mse);
}
//BIND_END

//BIND_METHOD TrainableSupervised use_dataset
//DOC_BEGIN
// float use_dataset(table params = { dataset input_dataset, dataset output_dataset })

/// Use a dataset with the mlp and puts outputs on other dataset

/// @param params.input_dataset A dataset object with each input pattern of the training.
/// @param params.output_dataset A dataset object with the corresponding output pattern for each input in input_dataset param.

//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "input_dataset",
		     "output_dataset",
		     0);
  
  DataSetFloat *inputds,*outputds;
  
  LUABIND_GET_TABLE_PARAMETER(1, input_dataset,
			      DataSetFloat, inputds);
  LUABIND_GET_TABLE_PARAMETER(1, output_dataset,
			      DataSetFloat, outputds);
  obj->useDataset(inputds, outputds);
}
//BIND_END
