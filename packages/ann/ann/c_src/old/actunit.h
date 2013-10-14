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
#ifndef ACTUNIT_H
#define ACTUNIT_H

#include "error_print.h"
#include "referenced.h"
#include "activation_function.h"
#include "gpu_mirrored_memory_block.h"
#include "ann_configuration.h"

namespace ANN {
  
  enum ActivationUnitsType {
    INPUTS_TYPE,
    HIDDEN_TYPE,
    OUTPUTS_TYPE
  };
  
  class ActivationUnits : public Referenced {
  protected:
    const ANNConfiguration &conf;
    ActivationUnitsType     type;
    unsigned int            fanin;
    bool is_sparse;
  public:
    float                   drop_factor;
    ActivationUnits(const ANNConfiguration &conf,
		    ActivationUnitsType type) :
      conf(conf), type(type), fanin(0), is_sparse(false), drop_factor(0.0f) {}
    virtual ~ActivationUnits() { }

    /// Method for getting the type of the activation units.
    ActivationUnitsType getType() const { return type; }

    /// Method for setting the type of the activation units.
    void setType(ActivationUnitsType type) { this->type=type; }

    /// Abstract method that returns the length of the array that represents
    /// the input (which is not necessarily equal to the number of neurons).
    virtual unsigned int size() const	 = 0;

    /// Abstract method that returns the number of neurons at the input.
    virtual unsigned int numNeurons() const = 0;

    /// Abstract method that returns a pointer to the array of units.
    virtual FloatGPUMirroredMemoryBlock *getPtr() = 0;

    /// Abstract method that returns a pointer to the array of errors.
    virtual FloatGPUMirroredMemoryBlock *getErrorVectorPtr() = 0;
    /// Devuelve un puntero al vector con la suma de los cuadrados de los pesos
    /// que entran a cada neurona
    virtual FloatGPUMirroredMemoryBlock *getSquaredLengthSums() = 0;
    /// devuelve el valor de offset, que sumado al size(), puede que el vector
    /// no comienze en 0
    virtual unsigned int getOffset() const = 0;

    /// Abstract method that returns a copy of the activation units.
    virtual ActivationUnits *clone(const ANNConfiguration &conf) = 0;

    /// Abstract method for resetting the activation units.
    virtual void reset(bool use_cuda) = 0;

    const ANNConfiguration &getConfReference() const { return conf; }
    /// for FAN IN computation
    virtual unsigned int getFanIn() const { return fanin; }
    virtual void increaseFanIn(unsigned int value) { fanin += value; }
    bool isSparse() const { return is_sparse; }
    void setSparse(bool v) { is_sparse = v; }
  };
  
  /// Implementa un vector de neuronas de tamanyo num_neurons *
  /// bunch_size, de manera que cada num_neurons tenemos una muestra, y
  /// tenemos tantas muestras como bunch.
  //  TODO: flag use_cuda para otros tipos de unidades de activacion.
  class RealActivationUnits : public ActivationUnits 
  {
    unsigned int		 num_neurons;
    FloatGPUMirroredMemoryBlock *activations;
    FloatGPUMirroredMemoryBlock *error_vector;
    FloatGPUMirroredMemoryBlock *squared_length_sums;
  public:
    RealActivationUnits(unsigned int        num_neurons,
			const ANNConfiguration &conf,
			ActivationUnitsType type,
			bool create_error_vector);
    ~RealActivationUnits();
    unsigned int		 size() const;
    FloatGPUMirroredMemoryBlock *getPtr();
    FloatGPUMirroredMemoryBlock *getErrorVectorPtr();
    FloatGPUMirroredMemoryBlock *getSquaredLengthSums();
    unsigned int     getOffset() const { return 0; }
    ActivationUnits *clone(const ANNConfiguration &conf);
    // deprecated:
    // const unsigned int &getBunchSize() const { return bunch_size; }
    unsigned int numNeurons() const {
      return size();
    }
    void reset(bool use_cuda);
    bool isSparse() const { return false; }
  };

  /// Implements a layer of neurons which activations are sparse. It is only
  /// valid as input, never as hidden or output layer. The activations array
  /// keeps a sequence of numbers 2N+1 numbers, being N the number of activated
  /// neurons:  N pos_1 val_1 pos_2 val_2 ... pos_N val_N, being pos_i and val_i
  /// the neuron identifier and its activation value.
  class SparseActivationUnits : public ActivationUnits 
  {
    unsigned int		 num_neurons;
    FloatGPUMirroredMemoryBlock *activations;
  public:
    SparseActivationUnits(unsigned int        num_neurons,
			  const ANNConfiguration &conf,
			  ActivationUnitsType type);
    ~SparseActivationUnits();
    unsigned int		 size() const;
    FloatGPUMirroredMemoryBlock *getPtr();
    FloatGPUMirroredMemoryBlock *getErrorVectorPtr();
    FloatGPUMirroredMemoryBlock *getSquaredLengthSums();
    unsigned int     getOffset() const { return 0; }
    ActivationUnits *clone(const ANNConfiguration &conf);
    // deprecated:
    // const unsigned int &getBunchSize() const { return bunch_size; }
    unsigned int numNeurons() const {
      return size();
    }
    void reset(bool use_cuda);
    bool isSparse() const { return true; }
  };

  /// implementa un subvector de neuronas que se encuentra como
  /// secuencia DENTRO de otro. Es util cuando hay matrices
  /// compartidas entre varias capas.
  class ActivationUnitsSlice : public ActivationUnits {
    ActivationUnits *units;
    unsigned int begin_unit, end_unit, num_units, num_neurons;
  public:
    ActivationUnitsSlice(ActivationUnits *units,
			 unsigned int begin_unit,
			 unsigned int end_unit,
			 const ANNConfiguration &conf,
			 ActivationUnitsType type);
    ~ActivationUnitsSlice();
    // aplica la funcion de activacion
    void activate(bool use_cuda);
    // devuelve el numero de unidades de entrada
    unsigned int size() const;
    // devuelve un puntero al vector de unidades, tantas unidades como
    // size()*bunch_size
    FloatGPUMirroredMemoryBlock *getPtr();
    // devuelve un puntero al vector con la suma de los errores
    FloatGPUMirroredMemoryBlock *getErrorVectorPtr();
    FloatGPUMirroredMemoryBlock *getSquaredLengthSums();
    // devuelve el valor de offset, que sumado al size(), es el valor
    // del major stride para CBLAS
    unsigned int getOffset() const;
    ActivationUnits *clone(const ANNConfiguration &conf);

    // deprecated:
    // const unsigned int &getBunchSize() const {
    //   return bunch_size;
    // }

    unsigned int numNeurons() const {
      return num_neurons;
    }
    void reset(bool use_cuda);
  };
}
#endif
