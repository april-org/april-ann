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

#include "datasetfloat_consumer.h"

namespace Functions {
  
  DataSetFloatConsumer::DataSetFloatConsumer(DataSetFloat *ds) : ds(ds) {
    IncRef(ds);
    ipat = 0;
  }
  
  DataSetFloatConsumer::~DataSetFloatConsumer() {
    DecRef(ds);
  }
  
  void DataSetFloatConsumer::put(float *pattern) {
    // check for a NULL pointer and check the number of patterns with the
    // counter value
    if (pattern != 0 && static_cast<int>(ipat) < ds->numPatterns())
      ds->putPattern(ipat++, pattern);
    // we have the property of pattern pointer, we delete it
    delete[] pattern;
  }
  
  void DataSetFloatConsumer::reset() {
    // with every reset, this consumer will overwrite the entire DataSetFloat
    ipat = 0;
  }
  
  void DataSetFloatConsumer::destroy() {
    // nothing more to do, only reset the counter
    ipat = 0;
  }

}
