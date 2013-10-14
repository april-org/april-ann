/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include "error_print.h"
#include "constants.h"
#include "ann.h"
#include "actunit.h"
#include "cblas_headers.h"
#include "gpu_helper.h"

#include "aux_hash_table.h" // required during cloning process
#include "hash_table.h"     // required during cloning process
using april_utils::hash;    // required during cloning process

namespace ANN {

  // construye un ANN vacio
  ANNBase::ANNBase(ANNConfiguration configuration) :
    conf(configuration) {
    total_num_inputs  = 0;
    total_num_outputs = 0;
    desired_output    = 0;
    output_neurons    = 0;
    output_errors     = 0;
    pattern_errors    = 0;
  }
  
  // lo destruye TODO
  ANNBase::~ANNBase() {
    for (unsigned int i=0; i<actions.size(); ++i)
      DecRef(actions[i]);
    for (unsigned int i=0; i<activations.size(); ++i)
      DecRef(activations[i]);
    for (unsigned int i=0; i<connections.size(); ++i)
      DecRef(connections[i]);
    delete desired_output;
    delete pattern_errors;
  }
  
  // registra un vector de activaciones como entrada
  void ANNBase::registerInput(ActivationUnits *actu) {
    inputs.push_back(actu);
    total_num_inputs += actu->size();
  }
  
  // registar un vector de activaciones como salida
  void ANNBase::registerOutput(ActivationUnits *actu) {
    if (output_neurons != 0)
      ERROR_EXIT(128, "Imposible to register more than 1 output layer!!!\n");
    output_neurons     = actu->getPtr();
    output_errors      = actu->getErrorVectorPtr();
    total_num_outputs += actu->size();
    desired_output     = new FloatGPUMirroredMemoryBlock(total_num_outputs*conf.max_bunch_size);
    pattern_errors     = new FloatGPUMirroredMemoryBlock(total_num_outputs*conf.max_bunch_size);
  }
  
  unsigned int ANNBase::getInputSize() const {
    return total_num_inputs;
  }

  unsigned int ANNBase::getOutputSize() const {
    return total_num_outputs;
  }

  // Funciones utiles para trabajar con la entrada y la salida
  
  // establece la entrada de la muestra 'bunch_pos' de la red neuronal
  // a partir del vector dado
  void ANNBase::setInput(const float *input, unsigned int bunch_pos) {
    unsigned int pos = 0;
    for (unsigned int i=0; i<inputs.size(); ++i) {
      const unsigned int sz  = inputs[i]->size();
      float *target_input    = inputs[i]->getPtr()->getPPALForWrite();
      cblas_scopy(sz, input + pos, 1, target_input + bunch_pos,
		  conf.max_bunch_size);
      pos += sz;
    }
  }
  
  // copia la salida calculada para la red con la muestra 'bunch_pos'
  // en el vector dado
  void ANNBase::copyOutput(float *output, unsigned int bunch_pos) {
    cblas_scopy(total_num_outputs, output_neurons->getPPALForRead() + bunch_pos,
		conf.max_bunch_size, output, 1);
  }
  
  // establece la salida deseada de la muestra 'bunch_pos', para poder
  // calcular con ella la funcion de error
  void ANNBase::setDesiredOutput(float *target_output, unsigned int bunch_pos) {
    cblas_scopy(total_num_outputs, target_output, 1,
		desired_output->getPPALForWrite() + bunch_pos,
		conf.max_bunch_size);
  }
  
  // inicializa a cero TODOS los vectores de error
  void ANNBase::resetErrorVectors() {
    // comento esto porque parece que da crash en ciertas redes...
    bool use_cuda_flag = conf.use_cuda_flag && (conf.max_bunch_size > 1);
    for (unsigned int i = 0; i < activations.size(); i++) {
      activations[i]->reset(use_cuda_flag);
    }
  }
  void ANNBase::resetPatternErrorsAuxiliarVector() {
    bool use_cuda_flag = conf.use_cuda_flag && (conf.max_bunch_size > 1);
    doVectorSetToZero(pattern_errors, total_num_outputs*conf.max_bunch_size,
		      1, 0, use_cuda_flag);
  }

  ///////////////////////////////////////////
  
  void ANNBase::setNumBunch(unsigned int num_bunch) 
  {
    // ojo, esto solamente cambia el current bunch, no el max_bunch_size
    // igual seria mejor cambiarle el nombre
    // FIXME: en este punto se tendria que validar que num_bunch < conf.max_bunch_size no?
    conf.cur_bunch_size = num_bunch;
  }

  // es necesario? deprecated? ojo, devuelve cur_bunch_size, no el max_bunch_size
  // quizas mejor desplegarlo en 2
  unsigned int ANNBase::getNumBunch()
  {
    return conf.cur_bunch_size;
  }

  void ANNBase::setUseCuda(bool use_cuda, bool pinned)
  {
#ifndef USE_CUDA
    if (use_cuda) {
      fprintf(stderr, "# Warning: Trying to set flag for using CUDA to true!\n");
      fprintf(stderr, "# Flag will be set to false. Check your script.\n");
      return;
    }
#else
    if (use_cuda) {
      GPUHelper::initHelper();
      if (pinned) {
	for (unsigned int i=0; i<inputs.size(); ++i)
	  inputs[i]->getPtr()->pinnedMemoryPageLock();
	output_neurons->pinnedMemoryPageLock();
	desired_output->pinnedMemoryPageLock();
      }
    }
#endif
    conf.use_cuda_flag = use_cuda;
  }

  bool ANNBase::getUseCuda()
  {
    return conf.use_cuda_flag;
  }

  // registar una capa de activaciones
  void ANNBase::registerActivationUnits(ActivationUnits *actu)
  {
    activations.push_back(actu);
    IncRef(actu);
  }
  
  // registar un objeto conexiones
  void ANNBase::registerConnections(Connections *conn)
  {
    connections.push_back(conn);
    IncRef(conn);
  }
  
  // registar una accion
  void ANNBase::registerAction(Action *action)
  {
    actions.push_back(action);
    IncRef(action);
  }
  
  // calcula la salida de la red neuronal dada una entrada
  bool ANNBase::calculate(const float *input_vector, unsigned int input_size,
			  float *output_vector, unsigned int output_size)
  {
    if (input_vector == 0 || output_vector == 0)
      ERROR_EXIT(119, "Vectors at calculate are null.\n");
    if (input_size != getInputSize())
      ERROR_EXIT(118, "Input sizes are different at calculate.\n");
    if (output_size != getOutputSize())
      ERROR_EXIT(117, "Output sizes are different at calculate.\n");

    if (input_vector == 0 || output_vector == 0)
      return false;
    else {
      unsigned int old_bunch_size = conf.cur_bunch_size;
      conf.cur_bunch_size = 1;
      setInput(input_vector, 0);
      doForward();
      copyOutput(output_vector, 0);
      conf.cur_bunch_size = old_bunch_size;
    }
    return true;
  }
  
  // Calcula la salida de la red neuronal a partir de un stream
  // (producer), y lo deja en otro stream (consumer). Trabajar en modo
  // stream permite aprovechar el modo bunch.
  void ANNBase::calculateInPipeline(Functions::FloatDataProducer *producer,
				    unsigned int input_size,
				    Functions::FloatDataConsumer *consumer,
				    unsigned int output_size)
  {
    float *input;
    unsigned int cur_bunch_pos  = 0;

    
    if (producer == 0)
      ERROR_EXIT(116, "Producer not created.\n");
    if (consumer == 0)
      ERROR_EXIT(115, "Consumer not created.\n");
    if (input_size != getInputSize())
      ERROR_EXIT(118, "Input sizes are different at pipeline.\n");
    if (output_size != getOutputSize())
      ERROR_EXIT(117, "Output sizes are different at pipeline.\n");

    while ((input = producer->get()) != 0) {
      setInput(input, cur_bunch_pos);
      delete[] input;
      ++cur_bunch_pos;
      if (cur_bunch_pos == conf.cur_bunch_size) {
	doForward();	
	for (unsigned int i = 0; i < conf.cur_bunch_size; ++i)  {
	  float *output = new float[output_size];
	  copyOutput(output, i);
	  consumer->put(output);
	}
	cur_bunch_pos = 0;
      }
    }
    if (cur_bunch_pos > 0) {
      unsigned int old_bunch_size = conf.cur_bunch_size;
      conf.cur_bunch_size = cur_bunch_pos;
      doForward();
      conf.cur_bunch_size = old_bunch_size;
      for (unsigned int i = 0; i < cur_bunch_pos; i++) {
	float *output = new float[output_size];
	copyOutput(output, i);
	consumer->put(output);
      }
    }
  }

  void ANNBase::cloneTopologyTo(ANNBase *other) {
    hash<void*,void*> clone_dict; // se destruye al final del metodo
    unsigned int current_input=0;
    for (unsigned int i=0; i<activations.size(); ++i) {
      ActivationUnits *actu = activations[i]->clone(other->getConfReference());
      clone_dict[(void*)activations[i]] = (void*)actu;
      other->registerActivationUnits(actu);
      if (current_input  <  inputs.size() &&
	  activations[i] == inputs[current_input])  {
	other->registerInput(actu);
	++current_input;
      }
      if (activations[i]->getPtr()==output_neurons) other->registerOutput(actu);
    }
    for (unsigned int i=0; i<connections.size(); ++i) {
      Connections *connect = connections[i]->clone();
      other->registerConnections(connect);
      clone_dict[(void*)connections[i]] = (void*)connect;
    }
    for (unsigned int i=0; i<actions.size(); ++i) {
      other->registerAction(actions[i]->clone(clone_dict,
					      other->getConfReference()));
    }
  }

  void ANNBase::pruneSubnormalAndCheckNormal() {
    for (unsigned int i=0; i<connections.size(); ++i)
      connections[i]->pruneSubnormalAndCheckNormal();
  }
  
  float ANNBase::getMaxWeights() const {
    // completar, he anyadido
    // virtual void getMaxWeights() = 0;
    // en connections
    // para que compile de momento:
    return 0;
  }

  Connections *ANNBase::getLayerConnections(unsigned int layer) {
    if (layer >= connections.size()) {
      ERROR_PRINT("Incorrect layer number!!!\n");
      exit(128);
    }
    return connections[layer];
  }
  
  ActivationUnits *ANNBase::getLayerActivations(unsigned int layer) {
    if (layer >= activations.size()) {
      ERROR_PRINT("Incorrect layer number!!!\n");
      exit(128);
    }
    return activations[layer];
  }

  Action *ANNBase::getAction(unsigned int idx) {
    if (idx >= actions.size())
      ERROR_EXIT(128, "Incorrect action number!!!\n");
    return actions[idx];
  }

  unsigned int ANNBase::copyWeightsTo(MatrixFloat **data,
				      MatrixFloat **old_data) const {
    const unsigned int numw = getNumWeights();
    *data     = new MatrixFloat(1, static_cast<int>(numw));
    *old_data = new MatrixFloat(1, static_cast<int>(numw));
    unsigned int pos = 0;
    for (unsigned int i=0; i<connections.size(); ++i)
      pos = connections[i]->copyWeightsTo(*data, *old_data, pos,
					  connections[i]->getNumInputs());
    return pos;
  }

}
