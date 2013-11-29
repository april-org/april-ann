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

#include "datasetfloat_producer.h"

namespace Functions {

  DataSetFloatProducer::DataSetFloatProducer(DataSetFloat *ds) : ds(ds) {
    IncRef(ds);
    ipat = 0;
  }
  
  DataSetFloatProducer::~DataSetFloatProducer() {
    DecRef(ds);
  }
  
  float *DataSetFloatProducer::get() {
    float *pattern = 0;
    // check the number of produced data vectors, if the maximum is achieved, a
    // NULL pointer will be returned
    if (static_cast<int>(ipat) < ds->numPatterns()) {
      pattern = new float[ds->patternSize()];
      ds->getPattern(ipat++, pattern);
    }
    return pattern;
  }
  
  void DataSetFloatProducer::reset() {
    ipat = 0;
  }
  
  void DataSetFloatProducer::destroy() {
    ipat = 0;
  }
}
