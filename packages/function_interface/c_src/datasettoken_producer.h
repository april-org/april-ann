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

#include "function_interface.h"
#include "datasetFloat.h"

namespace Functions {

  /// A specialization of FloatDataProducer which gets data from a dataset.
  /**
     A specialization of FloatDataProducer class which works as a source of data
     vectors. Data vectros are stored at the internal DataSetFloat
     attribute.
   */
  class DataSetFloatProducer : public FloatDataProducer{
    /// The internal DataSetFloat source.
    DataSetFloat *ds;
    /// An auxiliar counter which indicates the number of vectros produced.
    unsigned int  ipat;
  public:
    DataSetFloatProducer(DataSetFloat *ds);
    ~DataSetFloatProducer();
    float *get();
    void reset();
    void destroy();
  };
  
}
