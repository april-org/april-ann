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
#include "dice.h"
#include "trainsuper.h"
#include "datasetfloat_producer.h"
#include "datasetfloat_consumer.h"

namespace Trainable {

  float TrainableSupervised::trainOnePattern(Token *input,
					     Token *target_output) {
    beginTrainingBatch();
    trainPattern(input, target_output);
    return endTrainingBatch();
  }
  float TrainableSupervised::validateOnePattern(float *input,
						float *target_output) {
    beginValidateBatch();
    validatePattern(input, target_output);
    return endValidateBatch();
  }

  void TrainableSupervised::exitOnDatasetSizeError(DataSetFloat *input_dataset,
						   DataSetFloat *target_output_dataset) {
    if (input_dataset->patternSize() != static_cast<int>(getInputSize()))
      ERROR_EXIT2(128, "Incorrect patternSize!! InputSize=%u PatternSize=%d\n",
		  getInputSize(), input_dataset->patternSize());
    if (target_output_dataset->patternSize() != static_cast<int>(getOutputSize()))
      ERROR_EXIT2(128, "Incorrect patternSize!! OutputSize=%u PatternSize=%d\n",
		  getOutputSize(), target_output_dataset->patternSize());
  }

  /// Entrena con un dataset de entrada y otro de salida, admite ademas
  /// el parametro shuffle, que en caso de existir implica que el
  /// entrenamiento es aleatorio
  float TrainableSupervised::trainDataset(DataSetFloat	*input_dataset,
					  DataSetFloat	*target_output_dataset,
					  MTRand		*shuffle)
  {
    exitOnDatasetSizeError(input_dataset, target_output_dataset);
    float *input         = new float[getInputSize()];
    float *target_output = new float[getOutputSize()];
  
    int numberPatternsTrain      = input_dataset->numPatterns();
    const int threshold_big_perm = 65536;
  
    // Comenzamos a entrenar
    beginTrainingBatch();
    if (shuffle != 0) {
      // entrenamos con shuffle
      if (numberPatternsTrain <= threshold_big_perm) {
	int *vshuffle = new int[numberPatternsTrain];
	int index;
	shuffle->shuffle(numberPatternsTrain,vshuffle);
	for (int i = 0; i < numberPatternsTrain; i++) {
	  index = vshuffle[i];
	  input_dataset->getPattern(index, input);
	  target_output_dataset->getPattern(index, target_output);
	  // le damos el patron
	  trainPattern(input, target_output);
	}
	delete[] vshuffle;
      } else { // 2 level permutation:
	// numberPatternsTrain too big, we approximate the shuffle
	// permutation hierarchically (2 levels)
	const int log2divisor =  12;
	const int divisor     = 1 << log2divisor;
	int cociente   = numberPatternsTrain >> log2divisor;
	int *vshuffle1 = new int[divisor];
	int *vshuffle2 = new int[cociente+1];
	int index;
	shuffle->shuffle(divisor,vshuffle1);
	shuffle->shuffle(cociente+1,vshuffle2);
	for (int i = 0; i < divisor; i++) {
	  for (int j = 0; j <= cociente; j++) { 
	    index = (vshuffle2[j] << log2divisor) + vshuffle1[i];
	    if (index < numberPatternsTrain) { 
	      input_dataset->getPattern(index, input);
	      target_output_dataset->getPattern(index, target_output);
	      // le damos el patron
	      trainPattern(input, target_output);
	    }
	  }
	}
	delete[] vshuffle1;
	delete[] vshuffle2;
      } // end of 2 level permutation
    } else { // without shuffle
      for (int i = 0; i < numberPatternsTrain; i++) {
	input_dataset->getPattern(i, input);
	target_output_dataset->getPattern(i, target_output);
	trainPattern(input, target_output);
      }
    } // end of without shuffle
    float error = endTrainingBatch();
    delete[] input;
    delete[] target_output;
    return error;
  }

  float TrainableSupervised::
  trainDatasetWithReplacement(DataSetFloat	*input_dataset,
			      DataSetFloat	*target_output_dataset,
			      MTRand		*shuffle,
			      int			 replacement)
  {
    exitOnDatasetSizeError(input_dataset, target_output_dataset);
    float *input            = new float[getInputSize()];
    float *target_output    = new float[getOutputSize()];
    int numberPatternsTrain = input_dataset->numPatterns();
    // entrenamos:
    beginTrainingBatch();
    for (int i = 0; i < replacement; ++i) { 
      int index = shuffle->randInt(numberPatternsTrain-1);
      input_dataset->getPattern(index, input);
      target_output_dataset->getPattern(index, target_output);
      trainPattern(input, target_output);
    }
    float error = endTrainingBatch();
    delete[] input;
    delete[] target_output;
    return error;
  }

  float TrainableSupervised::
  trainDatasetWithDistribution(int num_classes,
			       DataSetFloat **input_datasets,
			       DataSetFloat **target_output_datasets,
			       double *aprioris,
			       MTRand *shuffle,
			       int replacement)
  {
    int *ds_sizes = new int[num_classes];
    for (int i=0; i<num_classes; i++) {
      exitOnDatasetSizeError(input_datasets[i],
			     target_output_datasets[i]);
      ds_sizes[i] = input_datasets[i]->numPatterns()-1;
    }
    float *input            = new float[getInputSize()];
    float *target_output    = new float[getOutputSize()];
    // creamos el random.dice
    dice *thedice = new dice(num_classes,aprioris);
    // entrenamos
    beginTrainingBatch();
    for (int i = 0; i < replacement; ++i) { 
      int whichclass = thedice->thrown(shuffle);
      int index = shuffle->randInt(ds_sizes[whichclass]);
      input_datasets[whichclass]->getPattern(index, input);
      target_output_datasets[whichclass]->getPattern(index, target_output);
      trainPattern(input, target_output);
    }
    // liberamos memoria:
    delete thedice;
    delete[] ds_sizes;
  
    float error = endTrainingBatch();
    delete[] input;
    delete[] target_output;
    return error;
  }

  // para validar con datasets
  float TrainableSupervised::validateDataset(DataSetFloat *input_dataset,
					     DataSetFloat *target_output_dataset)
  {
    exitOnDatasetSizeError(input_dataset,
			   target_output_dataset);
    float *input               = new float[getInputSize()];
    float *target_output       = new float[getOutputSize()];
    int numberPatternsValidate = input_dataset->numPatterns();
    beginValidateBatch();
    for (int i = 0; i < numberPatternsValidate; i++) { 
      input_dataset->getPattern(i, input);
      target_output_dataset->getPattern(i, target_output);
      validatePattern(input, target_output);
    }
    float error = endValidateBatch();
    delete[] input;
    delete[] target_output;
    return error;
  }

  float TrainableSupervised::
  validateDatasetWithReplacement(DataSetFloat *input_dataset,
				 DataSetFloat *target_output_dataset,
				 MTRand *shuffle,
				 int replacement)
  {
    exitOnDatasetSizeError(input_dataset,
			   target_output_dataset);
    float *input               = new float[getInputSize()];
    float *target_output       = new float[getOutputSize()];
    int numberPatternsValidate = input_dataset->numPatterns();
    beginValidateBatch();
    for (int i = 0; i < replacement; ++i) { 
      int index = shuffle->randInt(numberPatternsValidate);
      input_dataset->getPattern(index, input);
      target_output_dataset->getPattern(index, target_output);
      validatePattern(input, target_output);
    }
    float error = endValidateBatch();
    delete[] input;
    delete[] target_output;
    return error;
  }

  // para calcular la salida de la red con datasets
  void TrainableSupervised::useDataset(DataSetFloat *input_dataset,
				       DataSetFloat *output_dataset)
  {
    exitOnDatasetSizeError(input_dataset,
			   output_dataset);
    
    Functions::DataSetFloatProducer *producer;
    Functions::DataSetFloatConsumer *consumer;
    producer = new Functions::DataSetFloatProducer(input_dataset);
    consumer = new Functions::DataSetFloatConsumer(output_dataset);
    
    calculateInPipeline(producer, getInputSize(),
			consumer, getOutputSize());
    
    delete producer;
    delete consumer;
  }
  
}


/*
  void MLP::validate_dataset_with_weighted(DataSetFloat *input_dataset,
  DataSetFloat *output_dataset,
  DataSetFloat *weighted,
  float *output_suma_ponderaciones,
  int    num_ponderaciones,
  float *mask)
  {
  #ifdef BUNCH
  fprintf(stderr, "FUNCION NO PERMITIDA EN MODO BUNCH\n");
  exit(123);
  #endif
  int numberPatternsValidate = input_dataset->numPatterns();
  
  // proporcionamos otro dataset que indica la ponderación de cada
  // muestra presentada por tanto se trata de un dataset con el
  // mismo numPatterns que los input_dataset y output_dataset
  
  float *ponderaciones      = new float[num_ponderaciones];
  float *suma_ponderaciones = new float[num_ponderaciones];
  float *sum_sqr_err        = new float[num_ponderaciones]; // lo que devuelve
  for (int i=0; i < num_ponderaciones; i++) 
  sum_sqr_err[i] = suma_ponderaciones[i] = 0;
  for (int i = 0; i < numberPatternsValidate; i++) { 
  input_dataset->getPattern(i,vector);
  output_dataset->getPattern(i,desired_output);
  weighted->getPattern(i,ponderaciones);
  float msepattern = this->validate_pattern(mask);
  for (int j=0; j < num_ponderaciones; j++) {
  sum_sqr_err[j]        += ponderaciones[j]*msepattern;
  suma_ponderaciones[j] += ponderaciones[j];
  }
  }
  for (int i=0; i < num_ponderaciones; i++) {
  // printf("suma_ponderaciones[%d] da %lf\n",
  // i,suma_ponderaciones[i]);// PROVISIONAL
  if (suma_ponderaciones[i] > 0)
  output_suma_ponderaciones[i] = obtain_epoch_error(sum_sqr_err[i],
  suma_ponderaciones[i]);
  else output_suma_ponderaciones[i] = 0.0f;
  delete[] ponderaciones;
  delete[] suma_ponderaciones;
  delete[] sum_sqr_err;
  }
  }
*/
