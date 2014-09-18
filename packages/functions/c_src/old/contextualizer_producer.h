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

#ifndef CONTEXTUALIZER_PRODUCER_H
#define CONTEXTUALIZER_PRODUCER_H

#include "function_interface.h"
#include "context.h"

namespace Functions {

  /// A specialization of FloatDataProducer which gets data from another producer.
  /**
     A specialization of FloatDataProducer class which works as a source of data
     vectors. Data vectors are stored at the internal DoubleDataProducer
     attribute. A context is added to data vectors, in order to being used as
     inputs of classifiers, as for example a neural network. Also, this object
     could normalize the data vectors substracting the mean and dividing by the
     standard deviation.
  */
  class FloatContextualizerProducerFromDouble : public DataProducer<float> {
    /// Internal DoubleDataProducer source
    DataProducer<double> *orig_producer;
    /// Size of vectors at orig_producer
    int input_size;
    /// Size of vectros after context addition
    int output_size;
    /// Left context added to data vectors
    int left_context;
    /// Right context added to data vectors
    int right_context;
    /// Means vector, for mean normalization of data.
    float *means;
    /// Standard deviations vector, for variance normalization of data
    float *devs;
    /// Context object, it implements the contextualizer logic
    AprilUtils::context_of_vectors<float> ctxt;
  public:
    FloatContextualizerProducerFromDouble(DataProducer<double> *orig_producer,
					  int input_size,
					  int left_context,
					  int right_context,
					  float *means,
					  float *devs);
    ~FloatContextualizerProducerFromDouble();
    float *get();
    void reset();
  };
}

#endif // CONTEXTUALIZER_PRODUCER_H
