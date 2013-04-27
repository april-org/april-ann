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
#ifndef _TRAINABLE_SUPERVISED_H
#define _TRAINABLE_SUPERVISED_H

#include "loss_function.h"
#include "ann_component.h"
#include "function_interface.h"
#include "datasetToken.h"

namespace Trainable {
  
  /// Static class for improve lua functions efficiency
  class TrainableSupervised {
  public:
    static float trainDataset(ANNComponent *ann_component,
			      LossFunction *loss_function,
			      unsigned int  bunch_size,
			      DataSetToken *input_dataset,
			      DataSetToken *target_output_dataset,
			      MTRand       *shuffle=0);

    static float trainDatasetWithReplacement(ANNComponent *ann_component,
					     LossFunction *loss_function,
					     unsigned int  bunch_size,
					     DataSetToken *input_dataset,
					     DataSetToken *target_output_dataset,
					     MTRand *shuffle,
					     int replacement);

    static float trainDatasetWithDistribution(ANNComponent *ann_component,
					      LossFunction *loss_function,
					      unsigned int  bunch_size,
					      int num_classes,
					      DataSetToken **input_datasets,
					      DataSetToken **target_output_datasets,
					      double *aprioris,
					      MTRand *shuffle,
					      int replacement);
    
    static float validateDataset(ANNComponent *ann_component,
				 LossFunction *loss_function,
				 unsigned int  bunch_size,
				 DataSetToken *input_dataset,
				 DataSetToken *target_output_dataset);
    
    static float validateDatasetWithReplacement(ANNComponent *ann_component,
						LossFunction *loss_function,
						unsigned int  bunch_size,
						DataSetToken *input_dataset,
						DataSetToken *target_output_dataset,
						MTRand *shuffle,
						int replacement);
    
    static void useDataset(ANNComponent *ann_component,
			   unsigned int  bunch_size,
			   DataSetToken *input_dataset,
			   DataSetToken *output_dataset);
  };

}

#endif // _TRAINABLE_SUPERVISED_H
