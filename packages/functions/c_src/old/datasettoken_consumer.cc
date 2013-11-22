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

#include "datasettoken_consumer.h"

namespace Functions {
  
  DataSetTokenConsumer::DataSetTokenConsumer(DataSetToken *ds) : ds(ds) {
    IncRef(ds);
    ipat = 0;
  }
  
  DataSetTokenConsumer::~DataSetTokenConsumer() {
    DecRef(ds);
  }
  
  void DataSetTokenConsumer::put(Token *pattern) {
    // check for a NULL pointer and check the number of patterns with the
    // counter value
    if (pattern != 0 && static_cast<int>(ipat) < ds->numPatterns()) {
      IncRef(pattern);
      ds->putPattern(ipat++, pattern);
      DecRef(pattern);
    }
  }
  
  void DataSetTokenConsumer::reset() {
    // with every reset, this consumer will overwrite the entire DataSetToken
    ipat = 0;
  }
  
  void DataSetTokenConsumer::destroy() {
    // nothing more to do, only reset the counter
    ipat = 0;
  }

}
