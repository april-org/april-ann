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
#ifndef ANN_H
#define ANN_H
#include "aligned_memory.h"
#include "vector.h"
#include "constants.h"
#include "trainsuper.h"
#include "function_interface.h"
#include "actunit.h"
#include "connection.h"
#include "action.h"
#include "ann_configuration.h"
#include "activation_function.h"

using april_utils::vector;

namespace ANN {

  /// Clase base para definir objetos tipo ANN. Estos atributos son
  /// los minimos esperados:
  /// -- Vector de acciones, que permite ejecutar la red.
  /// -- Vector de activaciones, con las neuronas de cada capa.
  /// -- Vector de conexiones, con las conexiones entre capas
  /// -- Vector de entrada, salida deseada, salida, y errores a la salida.
  /// -- Tamanyo del bunch.
  /// -- Flag para el uso de CUDA
  class ANNBase : public Trainable::TrainableSupervised {
  protected:
    ANNConfiguration conf;
    FloatGPUMirroredMemoryBlock *output_neurons, *desired_output, *output_errors, *pattern_errors;
    unsigned int total_num_inputs, total_num_outputs;
    vector <ActivationUnits*> inputs;
    vector <ActivationUnits*> activations;
    vector <Connections*>     connections;
    vector <Action*>          actions;
    ANNBase(ANNConfiguration configuration);
    virtual void setNumBunch(unsigned int num_bunch);
    virtual unsigned int getNumBunch();
    virtual void doForward(bool during_training=false)  = 0;
    virtual void doBackward() = 0;
    
    void clearTopology() {
      delete desired_output;
      delete output_errors;
      for (unsigned int i=0; i<inputs.size(); ++i) DecRef(inputs[i]);
      for (unsigned int i=0; i<activations.size(); ++i) DecRef(activations[i]);
      for (unsigned int i=0; i<connections.size(); ++i) DecRef(connections[i]);
      for (unsigned int i=0; i<actions.size(); ++i) DecRef(actions[i]);
      inputs.clear();
      activations.clear();
      connections.clear();
      actions.clear();
    }
    
    float patternErrorsSum() {
      float  sum = 0.0f;
      const float *ptr = pattern_errors->getPPALForRead();
      for (unsigned int i=0; i<conf.max_bunch_size * total_num_outputs; ++i)
	sum += ptr[i];
      return sum;
    }
    
  public:
  
    virtual ~ANNBase();

    // DAG d'accions
  
    void registerActivationUnits(ActivationUnits *actu);
    void registerConnections(Connections *conn);
    void registerAction(Action *action);

    void registerInput(ActivationUnits *actu);
    void registerOutput(ActivationUnits *actu);
    
    /// This method converts output neurons into hidden neurons, and it sets output to zero
    void releaseOutput() {
      delete desired_output;
      delete pattern_errors;
      desired_output    = 0;
      pattern_errors    = 0;
      output_errors     = 0;
      output_neurons    = 0;
      total_num_outputs = 0;
      for (unsigned int i=0; i<activations.size(); ++i)
	if (activations[i]->getType() == OUTPUTS_TYPE)
	  activations[i]->setType(HIDDEN_TYPE);
    }

    unsigned int getInputSize() const;
    unsigned int getOutputSize() const;
  
    unsigned int getLayerConnectionsSize() const { return connections.size(); }
    unsigned int getLayerActivationsSize() const { return activations.size(); }
    
    Connections *getLayerConnections(unsigned int layer);
    ActivationUnits *getLayerActivations(unsigned int layer);
    Action          *getAction(unsigned int idx);

    virtual void setInput(const float *input, unsigned int bunch_pos);
    virtual void copyOutput(float *output, unsigned int bunch_pos);
    virtual void setDesiredOutput(float *target_output, unsigned int bunch_pos);
    void resetErrorVectors();
    void resetPatternErrorsAuxiliarVector();
  
    // inherited functions
    bool calculate(const float *input_vector, unsigned int input_size,
		   float *output_vector,      unsigned int output_size);
  
    void calculateInPipeline(Functions::FloatDataProducer *producer,
			     unsigned int input_size,
			     Functions::FloatDataConsumer *consumer,
			     unsigned int output_size);

    const ANNConfiguration &getConfReference() const { return conf; }
    ANNConfiguration        getConf()          const { return conf; }
    void cloneTopologyTo(ANNBase *other);


    void pruneSubnormalAndCheckNormal();

    virtual void setUseCuda(bool use_cuda, bool pinned);
    virtual bool getUseCuda();
    
    float getMaxWeights() const; // falta terminar
    
    unsigned int getNumWeights() const {
      unsigned int numw = 0;
      for (unsigned int i=0; i<connections.size(); ++i)
	numw += connections[i]->getNumWeights();
      return numw;
    }
    
    virtual unsigned int copyWeightsTo(MatrixFloat **data,
				       MatrixFloat **old_data) const;
    
  };
}
#endif // ANN_H
