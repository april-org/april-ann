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

#include "contextualizer_producer.h"

namespace Functions {

  FloatContextualizerProducerFromDouble::
  FloatContextualizerProducerFromDouble(DataProducer<double> *orig_producer,
					int input_size,
					int left_context,
					int right_context,
					float *means=0,
					float *devs =0) :
    orig_producer(orig_producer),
    input_size(input_size),
    left_context(left_context),
    right_context(right_context),
    means(means),
    devs(devs),
    ctxt(left_context,right_context,input_size)
  {
    output_size = input_size*(left_context+1+right_context);
    IncRef(orig_producer);
  }

  FloatContextualizerProducerFromDouble::~FloatContextualizerProducerFromDouble() {
    DecRef(orig_producer);
    delete[] means;
    delete[] devs;
  }

  float* FloatContextualizerProducerFromDouble::get() {
    float *resul = 0;
    // if contextualizer mode is end, shift sliding window
    if (ctxt.is_ended()) {
      ctxt.shift();
    } else {
      // we are not at the end, a new data vector will be pushed into
      // contextualizer object
      do {
	double *double_vec = orig_producer->get();
	float  *vec        = 0;
	if (double_vec != 0) {
	  vec = new float[input_size];
	  // normalize substracting mean and dividing by standard deviation
	  if (means!=0 && devs!=0)
	    for (int i=0; i<input_size; ++i)
	      vec[i] = (double_vec[i]-means[i])/devs[i];
	  else
	    for (int i=0; i<input_size; ++i)
	      vec[i] = float(double_vec[i]);
	  delete[] double_vec;
	}
	if (vec == 0) { // the last data vector has a special treatment
	  ctxt.end_input();
	} else {
	  ctxt.insert(vec);
	  delete[] vec;
	}
      } while (!ctxt.ready() && !ctxt.is_ended());
      if (ctxt.is_ended())
	ctxt.shift();
    }
    if (ctxt.ready()) {
      resul = ctxt.getOutputVector();
    }
    else if (ctxt.is_ended()) ctxt.reset();
    return resul;
  }

  void FloatContextualizerProducerFromDouble::reset() {
    ctxt.reset();
    orig_producer->reset();
  }
}
