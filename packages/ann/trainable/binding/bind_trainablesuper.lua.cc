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
#include "bind_ann_base.h"
//BIND_END

//BIND_HEADER_H
#include "datasetFloat.h"
#include "matrixFloat.h"
#include "trainsuper.h"

using namespace Trainable;

//BIND_END

//BIND_FUNCTION trainable.supervised_trainer_static_functions.train_dataset
//DOC_BEGIN
// float train_dataset(table params = { dataset input_dataset, dataset output_dataset, random shuffle = 0, replacement = input_dataset.numPatterns(), distribution = 0 })

/// Train the mlp with the parameters indicated in the table, and
/// returns the LOSS of the dataset.

/// @param params.input_dataset A dataset object with each input pattern of the training.
/// @param params.output_dataset A dataset object with the corresponding output pattern for each input in input_dataset param.

//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "ann",
		     "loss",
		     "bunch_size",
		     "input_dataset",
		     "output_dataset",
		     "shuffle",
		     "replacement",
		     "distribution",
		     0);
  
  float loss = 0;
  
  // 2 tipos de funcionamiento:
  // a) pasando información a priori sobre las distribuciones
  // b) pasando 2 datasets: entrada y salida
  
  ANNComponent *ann_component;
  LossFunction *loss_function;
  unsigned int bunch_size;
  
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNComponent, ann_component);
  LUABIND_GET_TABLE_PARAMETER(1, loss, LossFunction, loss_function);
  LUABIND_GET_TABLE_PARAMETER(1, bunch_size, uint, bunch_size);
  
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
    if (!mtrand) LUABIND_ERROR("train with distribution requires shuffle");
    int num_classes;
    LUABIND_TABLE_GETN(-1, num_classes);
    DataSetToken **v_ids = new DataSetToken*[num_classes];
    DataSetToken **v_ods = new DataSetToken*[num_classes];
    double *aprioris = new double[num_classes];
    for (int i=0; i<num_classes; i++) {
      // tomamos una tripleta
      lua_rawgeti(L, -1, i+1);
      check_table_fields(L, -2, "input_dataset", "output_dataset", "probability", 0);
      // comprobamos que tiene input, output, etc.
      LUABIND_GET_TABLE_PARAMETER(-2, input_dataset,
				  DataSetToken, v_ids[i]);
      LUABIND_GET_TABLE_PARAMETER(-2, output_dataset,
				  DataSetToken, v_ods[i]);
      LUABIND_GET_TABLE_PARAMETER(-2, probability,
				  double, aprioris[i]);
      lua_pop(L,1); // nos cargamos la tabla i-ésima
    }
    loss = obj->trainDatasetWithDistribution(ann_component,
					     loss_function,
					     bunch_size,
					     num_classes,
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
    
    DataSetToken *inputds,*outputds;
    
    LUABIND_GET_TABLE_PARAMETER(1, input_dataset,
				DataSetToken, inputds);
    LUABIND_GET_TABLE_PARAMETER(1, output_dataset,
				DataSetToken, outputds);
    
    if (mtrand != 0 && replacement != 0) { // shuffle with replacement
      loss = obj->trainDatasetWithReplacement(ann_component,
					      loss_function,
					      bunch_size,
					      inputds,
					      outputds,
					      mtrand,
					      replacement);
    }
    else { // without replacement
      loss = obj->trainDataset(ann_component, loss_function, bunch_size,
			       inputds, outputds, mtrand);
    }
  } // end of using input and output dataset
  LUABIND_RETURN(float, loss);
}
//BIND_END


//BIND_FUNCTION trainable.supervised_trainer_static_functions.validate_dataset
//DOC_BEGIN
// float validate_dataset(table params = { dataset input_dataset, dataset output_dataset, random shuffle = 0, replacement = input_dataset.numPatterns(), distribution = 0 })

/// Validate a dataset with the mlp and the parameters indicated in the table, and
/// returns the LOSS of the dataset.

/// @param params.input_dataset A dataset object with each input pattern of the training.
/// @param params.output_dataset A dataset object with the corresponding output pattern for each input in input_dataset param.

//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1,
		     "ann",
		     "loss",
		     "bunch_size",
		     "input_dataset",
		     "output_dataset",
		     "shuffle",
		     "replacement",
		     0);

  ANNComponent *ann_component;
  LossFunction *loss_function;
  unsigned int bunch_size;

  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNComponent, ann_component);
  LUABIND_GET_TABLE_PARAMETER(1, loss, LossFunction, loss_function);
  LUABIND_GET_TABLE_PARAMETER(1, bunch_size, uint, bunch_size);
  
  float loss = 0;
  
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
  
  DataSetToken *inputds,*outputds;
  
  LUABIND_GET_TABLE_PARAMETER(1, input_dataset,
			      DataSetToken, inputds);
  LUABIND_GET_TABLE_PARAMETER(1, output_dataset,
			      DataSetToken, outputds);
    
  if (replacement != 0) { // shuffle with replacement
    loss = obj->validateDatasetWithReplacement(ann_component,
					       loss_function,
					       bunch_size,
					       inputds,
					       outputds,
					       mtrand,
					       replacement);
  }
  else { // without replacement
    loss = obj->validateDataset(ann_component, loss_function, bunch_size,
				inputds, outputds);
  } // end of using input and output dataset
  LUABIND_RETURN(float, loss);
}
//BIND_END

//BIND_FUNCTION trainable.supervised_trainer_static_functions.use_dataset
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
		     "ann",
		     "bunch_size",
		     "input_dataset",
		     "output_dataset",
		     0);

  ANNComponent *ann_component;
  unsigned int bunch_size;
  
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNComponent, ann_component);
  LUABIND_GET_TABLE_PARAMETER(1, bunch_size, uint, bunch_size);
  
  DataSetToken *inputds,*outputds;
  
  LUABIND_GET_TABLE_PARAMETER(1, input_dataset,
			      DataSetToken, inputds);
  LUABIND_GET_TABLE_PARAMETER(1, output_dataset,
			      DataSetToken, outputds);
  obj->useDataset(ann_component, bunch_size, inputds, outputds);
}
//BIND_END
