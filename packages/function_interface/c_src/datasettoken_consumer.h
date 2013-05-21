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
#include "datasetToken.h"

namespace Functions {

  /// A specialization of DataConsumer which put data into a dataset.
  /**
     A specialization of FloatDataConsumer class which works as a sink of data
     vectors. Data vectros will be stored in the internal DataSetToken
     attribute. The DataSetToken must have enough space for all consumed
     vectors.
   */
  class DataSetTokenConsumer : public DataConsumer{
    /// Internal DataSetToken where tokens will be stored.
    DataSetToken *ds;
    /// Auxiliar iterator counter of how many vectors were processed.
    unsigned int  ipat;
  public:
    DataSetTokenConsumer(DataSetToken *ds);
    ~DataSetTokenConsumer();
    void put(Token *pattern);
    void reset();
    void destroy();
  };
  
}
