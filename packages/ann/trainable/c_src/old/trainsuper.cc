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
// #include "datasetfloat_producer.h"
// #include "datasetfloat_consumer.h"

namespace Trainable {
  
  void trainPattern(ANNComponent *ann_component,
		    LossFunction *loss_function,
		    Token        *input,
		    Token        *target) {
    ann_component->reset();
    Token *output = ann_component->doForward(input);
    loss_function->addLoss(output, target);
    ann_component->doBackprop(loss_function->computeGrandient(output, target));
    ann_component->doUpdate();
  }

  void validatePattern(ANNComponent *ann_component,
		       LossFunction *loss_function,
		       Token        *input,
		       Token        *target) {
    ann_component->reset();
    Token *output = ann_component->doForward(input);
    loss_function->addLoss(output, target);
  }
  
  /// Entrena con un dataset de entrada y otro de salida, admite ademas
  /// el parametro shuffle, que en caso de existir implica que el
  /// entrenamiento es aleatorio
  static float TrainableSupervised::trainDataset(ANNComponent	*ann_component,
						 LossFunction	*loss_function,
						 unsigned int	 bunch_size,
						 DataSetToken	*input_dataset,
						 DataSetToken	*target_output_dataset,
						 MTRand	*shuffle)
  {
    exitOnDatasetSizeError(input_dataset, target_output_dataset);
    
    int numberPatternsTrain      = input_dataset->numPatterns();
    const int threshold_big_perm = 65536;
    
    TokenBunchVector *input_bunch  = new TokenBunchVector(bunch_size);
    TokenBunchVector *output_bunch = new TokenBunchVector(bunch_size);
    input_bunch->clear();
    target_bunch->clear();
    
    // Comenzamos a entrenar
    if (shuffle != 0) {
      // entrenamos con shuffle
      if (numberPatternsTrain <= threshold_big_perm) {
	int *vshuffle = new int[numberPatternsTrain];
	int index;
	shuffle->shuffle(numberPatternsTrain,vshuffle);
	for (int i = 0; i < numberPatternsTrain; i++) {
	  index = vshuffle[i];
	  Token *input  = input_dataset->getPattern(index);
	  Token *target = target_output_dataset->getPattern(index);
	  input_bunch->push_back(input);
	  target_bunch->push_back(target);
	  if (input_bunch->size() == bunch_size) {
	    trainPattern(ann_component, loss_function,
			 input_bunch, target_bunch);
	    input_bunch->clear();
	    target_bunch->clear();
	  }
	}
	if (input_bunch->size() > 0) {
	  trainPattern(ann_component, loss_function,
		       input_bunch, target_bunch);
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
	      Token *input  = input_dataset->getPattern(index);
	      Token *target = target_output_dataset->getPattern(index);
	      input_bunch->push_back(input);
	      target_bunch->push_back(target);
	      if (input_bunch->size() == bunch_size) {
		trainPattern(ann_component, loss_function,
			     input_bunch, target_bunch);
		input_bunch->clear();
		target_bunch->clear();
	      }
	    }
	  }
	  if (input_bunch->size() > 0) {
	    trainPattern(ann_component, loss_function,
			 input_bunch, target_bunch);
	    input_bunch->clear();
	    target_bunch->clear();
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
    float error = loss_function->getAccumLoss();
    delete input_bunch;
    delete target_bunch;
    return error;
  }

  static float TrainableSupervised::
  trainDatasetWithReplacement(ANNComponent	*ann_component,
			      LossFunction	*loss_function,
			      unsigned int	 bunch_size,
			      DataSetToken	*input_dataset,
			      DataSetToken	*target_output_dataset,
			      MTRand		*shuffle,
			      int		 replacement)
  {
    exitOnDatasetSizeError(input_dataset, target_output_dataset);
    int numberPatternsTrain = input_dataset->numPatterns();

    TokenBunchVector *input_bunch  = new TokenBunchVector(bunch_size);
    TokenBunchVector *output_bunch = new TokenBunchVector(bunch_size);
    input_bunch->clear();
    target_bunch->clear();
    
    for (int i = 0; i < replacement; ++i) { 
      int index = shuffle->randInt(numberPatternsTrain-1);
      Token *input  = input_dataset->getPattern(index);
      Token *target = target_output_dataset->getPattern(index);
      input_bunch->push_back(input);
      target_bunch->push_back(target);
      if (input_bunch->size() == bunch_size) {
	trainPattern(ann_component, loss_function,
		     input_bunch, target_bunch);
	input_bunch->clear();
	target_bunch->clear();
      }
    }
    if (input_bunch->size() > 0)
      trainPattern(ann_component, loss_function,
		   input_bunch, target_bunch);
    float error = loss_function->getAccumLoss();
    delete input_bunch;
    delete target_bunch;
    return error;
  }

  static float TrainableSupervised::
  trainDatasetWithDistribution(ANNComponent	 *ann_component,
			       LossFunction	 *loss_function,
			       unsigned int	  bunch_size,
			       int		  num_classes,
			       DataSetToken	**input_datasets,
			       DataSetToken	**target_output_datasets,
			       double		 *aprioris,
			       MTRand		 *shuffle,
			       int		  replacement)
  {
    int *ds_sizes = new int[num_classes];
    for (int i=0; i<num_classes; i++) {
      exitOnDatasetSizeError(input_datasets[i],
			     target_output_datasets[i]);
      ds_sizes[i] = input_datasets[i]->numPatterns()-1;
    }

    TokenBunchVector *input_bunch  = new TokenBunchVector(bunch_size);
    TokenBunchVector *output_bunch = new TokenBunchVector(bunch_size);
    input_bunch->clear();
    target_bunch->clear();
    
    // creamos el random.dice
    dice *thedice = new dice(num_classes,aprioris);
    for (int i = 0; i < replacement; ++i) { 
      int whichclass = thedice->thrown(shuffle);
      int index = shuffle->randInt(ds_sizes[whichclass]);
      Token *input  = input_datasets[whichclass]->getPattern(index);
      Token *target = target_output_datasets[whichclass]->getPattern(index);
      input_bunch->push_back(input);
      target_bunch->push_back(target);
      if (input_bunch->size() == bunch_size) {
	trainPattern(ann_component, loss_function,
		     input_bunch, target_bunch);
	input_bunch->clear();
	target_bunch->clear();
      }
    }
    if (input_bunch->size() > 0) {
      trainPattern(ann_component, loss_function,
		   input_bunch, target_bunch);
      input_bunch->clear();
      target_bunch->clear();
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
  static float TrainableSupervised::validateDataset(ANNComponent	*ann_component,
						    LossFunction	*loss_function,
						    unsigned int	 bunch_size,
						    DataSetToken	*input_dataset,
						    DataSetToken	*target_output_dataset)
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

  static float TrainableSupervised::
  validateDatasetWithReplacement(ANNComponent	*ann_component,
				 LossFunction	*loss_function,
				 unsigned int	 bunch_size,
				 DataSetToken	*input_dataset,
				 DataSetToken	*target_output_dataset,
				 MTRand		*shuffle,
				 int		 replacement)
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
  static void TrainableSupervised::useDataset(ANNComponent	*ann_component,
					      unsigned int	 bunch_size,
					      DataSetToken	*input_dataset,
					      DataSetToken	*output_dataset)
  {
    /*
      exitOnDatasetSizeError(input_dataset,
      output_dataset);
      
      Functions::DataSetTokenProducer *producer;
      Functions::DataSetTokenConsumer *consumer;
      producer = new Functions::DataSetTokenProducer(input_dataset);
      consumer = new Functions::DataSetTokenConsumer(output_dataset);
      
      calculateInPipeline(producer, getInputSize(),
      consumer, getOutputSize());
      
      delete producer;
      delete consumer;
    */
  }
  
}
