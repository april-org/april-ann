/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#include "utilMatrixIO.h"

namespace basics {
  
  april_utils::constString readULine(april_utils::UniquePtr<april_io::StreamInterface> &stream,
                                     april_utils::UniquePtr<april_io::CStringStream> &dest) {
    // Not needed, it is done in extractULineFromStream: dest->clear(); 
    extractULineFromStream(stream.get(), dest.get());
    return dest->getConstString();
  }
  
} // namespace basics
