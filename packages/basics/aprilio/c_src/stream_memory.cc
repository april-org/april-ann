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
#include "stream_memory.h"

namespace AprilIO {
  
  size_t extractLineFromStream(StreamInterface *source, StreamMemory *dest,
                               bool keep_delim) {
    return source->get(dest, "\n\r", keep_delim);
  }

  size_t extractULineFromStream(StreamInterface *source, StreamMemory *dest,
                                bool keep_delim) {
    do {
      dest->clear();
      source->get(dest, "\n\r", keep_delim);
    } while ( ((dest->size() > 0) && ((*dest)[0] == '#')) ||
              // ignore empty lines
              ((dest->size() == 1) && ((*dest)[0] == '\r' || (*dest)[0] == '\n')) );
    return dest->size();
  }
  
} // namespace AprilIO
