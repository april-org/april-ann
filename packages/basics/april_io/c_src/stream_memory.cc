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

namespace april_io {
  
  char StreamMemory::DUMMY_CHAR = '\0';
  const char *StreamMemory::NO_ERROR_STRING = "No error";

  size_t extractLineFromStream(Stream *source, StreamMemory *dest) {
    return source->get(dest, "\n\r");
  }

  size_t extractULineFromStream(Stream *source, StreamMemory *dest) {
    do {
      dest->clear();
      source->get(dest, "\n\r");
    } while ((dest->size() > 0) && ((*dest)[0] == '#'));
    return dest->size();
  }
  
} // namespace april_io
