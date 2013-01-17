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
#ifndef ACTION_H
#define ACTION_H

#include "actunit.h"
#include "connection.h"
#include "referenced.h"
#include "error_print.h"
#include "ann_configuration.h"
#include "aux_hash_table.h" // required during cloning process
#include "hash_table.h"     // required during cloning process
using april_utils::hash;    // required during cloning process

namespace ANN {

  class Action : public Referenced {
  protected:
    const ANNConfiguration &conf;
  public:
    Action(const ANNConfiguration &conf) : conf(conf) { }
    // Wrapper of connections->forward(....)
    virtual ~Action() { }
    virtual void doForward() = 0;
    virtual void doBackward() = 0;
    virtual Action *clone(hash<void*,void*> &clone_dict,
			  const ANNConfiguration &conf) = 0;
    // Changes the value of a training parameter
    virtual void setOption(const char *name, double value) { }
    virtual bool hasOption(const char *name) { return false; }
    virtual double getOption(const char *name) { return 0.0; }
  };
  
  // template<typename T>
  // class SharedAction : public Action {
  //   ActivationUnits      *inputs;
  //   ActivationUnits      *outputs;
  //   Connections          *shared_weights;
  //   unsigned int          input_slice_size;
  //   unsigned int          output_slice_size;
  //   unsigned int          num_slices;
  //   Action              **actions;

  // public:
  //   SharedAction(ActivationUnits *inputs,
  // 		 ActivationUnits *outputs,
  // 		 Connections     *weights,
  // 		 unsigned int input_slice_size,
  // 		 unsigned int output_slice_size) :
  //     inputs(inputs), outputs(outputs), shared_weights(weights),
  //     input_slice_size(input_slice_size), output_slice_size(output_slice_size) {
  //     if (inputs->size() % input_slice_size != 0) {
  // 	ERROR_PRINT("Incorrect input slice size!!!");
  // 	exit(128);
  //     }
  //     if (outputs->size() % output_slice_size != 0) {
  // 	ERROR_PRINT("Incorrect output slice size!!!");
  // 	exit(128);
  //     }
  //     num_slices = inputs->size()  / input_slice_size;
  //     unsigned int num_output_slices = outputs->size() / output_slice_size;
  //     if (num_slices != num_output_slices) {
  // 	ERROR_PRINT("Input and output number of slices must be equals");
  // 	exit(128);
  //     }
      
  //     actions = new Action*[num_slices];
  //     for (unsigned int i=0; i<num_slices; ++i) {
  // 	ActivationUnitsSlice *input_slice =
  // 	  new ActivationUnitsSlice(inputs,
  // 				   input_slice_size*i,
  // 				   input_slice_size*(i+1)-1,
  // 				   inputs->getConfReference());
  // 	ActivationUnitsSlice *output_slice =
  // 	  new ActivationUnitsSlice(outputs,
  // 				   output_slice_size*i,
  // 				   output_slice_size*(i+1)-1,
  // 				   outputs->getConfReference());
  // 	actions[i] = new T(input_slice, output_slice, shared_weights);
  //     }

  //     IncRef(inputs);
  //     IncRef(outputs);
  //     IncRef(shared_weights);
  //   }
  //   ~SharedAction() {
  //     for (unsigned int i=0; i<num_slices; ++i) delete actions[i];
  //     delete[] actions;
  //     DecRef(inputs);
  //     DecRef(outputs);
  //     DecRef(shared_weights);
  //   }
  //   void doForward() {
  //     for (unsigned int i=0; i<num_slices; ++i) actions[i]->doForward();
  //   }
  //   void doBackward() {
  //     for (unsigned int i=0; i<num_slices; ++i) actions[i]->doBackward();
  //   }
  //   Action *clone(hash<void*,void*> &clone_dict,
  // 		  const ANNConfiguration &conf) {
  //     return new SharedAction((ActivationUnits *)clone_dict[inputs],
  // 			      (ActivationUnits *)clone_dict[outputs],
  // 			      (Connections *)clone_dict[shared_weights],
  // 			      input_slice_size,
  // 			      output_slice_size);
  //   }
  // };

  class CopyAction : public Action {
    ActivationUnits *inputs;
    ActivationUnits *outputs;
    // Starts at input_first_unit, copy_size length, and copies it to output_first_unit
    unsigned int     first_input_unit;
    unsigned int     first_output_unit;
    unsigned int     copy_size;
  public:
    CopyAction(ActivationUnits *inputs, ActivationUnits *outputs,
	       unsigned int first_input_unit, unsigned int first_output_unit,
	       unsigned int copy_size);
    ~CopyAction();
    void doForward();
    void doBackward();
    Action *clone(hash<void*,void*> &clone_dict,		  
		  const ANNConfiguration &conf) { return 0; }
  };
}

#endif // ACTION_H
