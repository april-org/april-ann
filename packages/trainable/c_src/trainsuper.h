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

#include "function_interface.h"
#include "datasetToken.h"

#define mSetOption(var_name,var) if(!strcmp(name,(var_name))){(var)=value;return;}
#define mHasOption(var_name) if(!strcmp(name,(var_name))) return true;
#define mGetOption(var_name, var) if(!strcmp(name,(var_name)))return (var);

namespace Trainable {

  class TrainableSupervised : public Functions::FloatFloatFunctionInterface {
  
    void exitOnDatasetSizeError(DataSetToken *input_dataset,
				DataSetToken *output_dataset);

  protected:
    /// Entre begin y end va el conjunto de patrones de una epoca
    virtual void beginTrainingBatch() = 0;
    // le da un patrin para entrenar
    virtual void trainPattern(float *input, float *target_output)    = 0;
    virtual float endTrainingBatch()   = 0;

    virtual void beginValidateBatch() = 0;
    // le da un patron para calcular el error, PERO sin entrenar
    virtual void validatePattern(float *input, float *target_output) = 0;
    virtual float endValidateBatch()   = 0;
  
  public:
    
    virtual ~TrainableSupervised() { }

    virtual void setOption(const char *name, double value) = 0;
    virtual bool hasOption(const char *name)               = 0;
    virtual double getOption(const char *name)             = 0;

    virtual void loadModel(const char *filename) = 0;
    virtual void saveModel(const char *filename) = 0;
    
    // Heredados de FunctionBase
    // virtual unsigned int getInputSize()  const = 0;
    // virtual unsigned int getOutputSize() const = 0;
    Functions::NS_function_io::type getInputType() const {
      return Functions::NS_function_io::FLOAT;
    }
    Functions::NS_function_io::type getOutputType() const {
      return Functions::NS_function_io::FLOAT;
    }
    
    ////////// METODOS IMPLEMENTADOS, SE PERMITE SOBREESCRIBIRLOS ///////////
    // Metodos ya implementados
    virtual float trainOnePattern(Token *input, Token *target_output);
    virtual float validateOnePattern(Token *input, Token *target_output);

    // para entrenar con datasets
    virtual float trainDataset(DataSetToken *input_dataset,
			       DataSetToken *target_output_dataset,
			       MTRand       *shuffle=0);
    virtual float trainDatasetWithReplacement(DataSetToken *input_dataset,
					      DataSetToken *target_output_dataset,
					      MTRand *shuffle,
					      int replacement);
    virtual float trainDatasetWithDistribution(int num_classes,
					       DataSetToken **input_datasets,
					       DataSetToken **target_output_datasets,
					       double *aprioris,
					       MTRand *shuffle,
					       int replacement);
    
    // para validar con datasets
    virtual float validateDataset(DataSetToken *input_dataset,
				  DataSetToken *target_output_dataset);
    virtual float validateDatasetWithReplacement(DataSetToken *input_dataset,
						 DataSetToken *target_output_dataset,
						 MTRand *shuffle,
						 int replacement);
    virtual void useDataset(DataSetToken *input_dataset,
			    DataSetToken *output_dataset);
  };

}

#endif // _TRAINABLE_SUPERVISED_H
