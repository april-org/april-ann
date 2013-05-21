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
#include <cstdlib>
#include "aligned_memory.h"
#include "actunit.h"
#include "constants.h"
#include "error_print.h"
#include "wrapper.h"

namespace ANN {
  
  // Layer composed of units with a Real type
  // Each neuron takes a value within the range of the real numbers
  RealActivationUnits::RealActivationUnits(unsigned int num_neurons,
					   const ANNConfiguration &conf,
					   ActivationUnitsType type,
					   bool create_error_vector) :
    ActivationUnits(conf, type),
    num_neurons(num_neurons) {

    activations = new FloatGPUMirroredMemoryBlock(num_neurons * conf.max_bunch_size);
    if (activations == 0)
      ERROR_EXIT(141, "Can not alloc unit vectors\n");
    if (create_error_vector) {
      error_vector = new FloatGPUMirroredMemoryBlock(num_neurons * conf.max_bunch_size);
      if (error_vector == 0)
	ERROR_EXIT(141, "Can not alloc error vectors\n");
    }
    else error_vector = 0;
    
    squared_length_sums = 0;
  }

  RealActivationUnits::~RealActivationUnits() {
    delete activations;
    if (error_vector) delete error_vector;
    if (squared_length_sums) delete squared_length_sums;
  }
  
  void RealActivationUnits::reset(bool use_cuda) {
    /* ATTENTION: Be careful with this, resetting the input layer causes
       the inputs to be 0, so the training is not performed in a correct way.
    */
    if (error_vector != 0) {
      doVectorSetToZero(activations, num_neurons*conf.max_bunch_size, 1, 0,
			use_cuda);
      doVectorSetToZero(error_vector, num_neurons*conf.max_bunch_size, 1, 0,
			use_cuda);
    }
    if (squared_length_sums)
      doVectorSetToZero(squared_length_sums, num_neurons, 1, 0, use_cuda);
  }
  
  ActivationUnits *RealActivationUnits::clone(const ANNConfiguration &conf) {
    RealActivationUnits *units = new RealActivationUnits(num_neurons,
							 conf,
							 type,
							 error_vector!=0);
    units->is_sparse = is_sparse;
    return units;
  }
  
  unsigned int RealActivationUnits::size() const {
    return num_neurons;
  }

  FloatGPUMirroredMemoryBlock *RealActivationUnits::getPtr() {
    return activations;
  }

  FloatGPUMirroredMemoryBlock *RealActivationUnits::getErrorVectorPtr() {
    return error_vector;
  }

  FloatGPUMirroredMemoryBlock *RealActivationUnits::getSquaredLengthSums() {
    if (squared_length_sums == 0)
      squared_length_sums = new FloatGPUMirroredMemoryBlock(num_neurons);
    return squared_length_sums;
  }

  /////////////////////////////////////////////////////////////////////

  // Layer composed of units with a Real type
  // Each neuron takes a value within the range of the real numbers
  SparseActivationUnits::SparseActivationUnits(unsigned int num_neurons,
					       const ANNConfiguration &conf,
					       ActivationUnitsType type) :
    ActivationUnits(conf, type),
    num_neurons(num_neurons) {

    activations = new FloatGPUMirroredMemoryBlock((2 * num_neurons + 1) *
						  conf.max_bunch_size);
    if (activations == 0)
      ERROR_EXIT(141, "Can not alloc unit vectors\n");
  }

  SparseActivationUnits::~SparseActivationUnits() {
    delete activations;
  }
  
  void SparseActivationUnits::reset(bool use_cuda) {
  }
  
  ActivationUnits *SparseActivationUnits::clone(const ANNConfiguration &conf) {
    SparseActivationUnits *units = new SparseActivationUnits(num_neurons,
							     conf,
							     type);
    return units;
  }
  
  unsigned int SparseActivationUnits::size() const {
    return 2*num_neurons + 1;
  }

  FloatGPUMirroredMemoryBlock *SparseActivationUnits::getPtr() {
    return activations;
  }

  FloatGPUMirroredMemoryBlock *SparseActivationUnits::getErrorVectorPtr() {
    return 0;
  }

  FloatGPUMirroredMemoryBlock *SparseActivationUnits::getSquaredLengthSums() {
    return 0;
  }

  /////////////////////////////////////////////////////////////////////


  /*  
      class ActivationUnitsSlice : public ActivationUnits {
      ActivationUnits *units;
      unsigned int begin_unit, end_unit, num_units;
      const unsigned int &bunch_size;
      public:
  */
  ActivationUnitsSlice::ActivationUnitsSlice(ActivationUnits *units,
					     unsigned int begin_unit,
					     unsigned int end_unit,
					     const ANNConfiguration &conf,
					     ActivationUnitsType type) :
    ActivationUnits(conf, type),
    units(units), begin_unit(begin_unit), end_unit(end_unit),
    num_units(end_unit - begin_unit + 1) {
    IncRef(units);
    num_neurons = units->numNeurons() / units->size() * num_units;
  }

  ActivationUnitsSlice::~ActivationUnitsSlice() {
    DecRef(units);
  }

  // Returns the number of units
  unsigned int ActivationUnitsSlice::size() const {
    return num_units;
  }

  // Returns a pointer to the units vector, whose is size equal to
  // size()*bunch_size
  FloatGPUMirroredMemoryBlock *ActivationUnitsSlice::getPtr() {
    return units->getPtr();
  }

  // Returns a pointer to the error vector
  FloatGPUMirroredMemoryBlock *ActivationUnitsSlice::getErrorVectorPtr() {
    return units->getErrorVectorPtr();
  }

  FloatGPUMirroredMemoryBlock *ActivationUnitsSlice::getSquaredLengthSums() {
    return units->getSquaredLengthSums();
  }

  // Returns the value of the offset. If we add it to size(), the value obtained
  // is the major stride for CBLAS
  // TODO: This might not be true after the change to Column Major.
  unsigned int ActivationUnitsSlice::getOffset() const {
    return begin_unit;
  }

  ActivationUnits *ActivationUnitsSlice::clone(const ANNConfiguration &conf) {
    ERROR_PRINT("Impossible to clone ActivationUnitsSlice class!!!\n");
    exit(128);
    return 0;
  }
  
  void ActivationUnitsSlice::reset(bool use_cuda) {
    units->reset(use_cuda);
  }
}
