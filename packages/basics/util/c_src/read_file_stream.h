/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef READ_FILE_STREAM_H
#define READ_FILE_STREAM_H

#include "constString.h"

class ReadFileStream {
  char *buffer;
  int max_buffer_len, buffer_pos, buffer_len;
  FILE *f;
  
  bool moveAndFillBuffer();
  bool resizeAndFillBuffer();
  bool trim(const char *delim);

public:
  ReadFileStream(const char *path);
  ~ReadFileStream();
  constString getToken(const char *delim);
  
  // para ser compatible con el interfaz de constString
  constString extract_line();
  constString extract_u_line();
};

#endif // READ_FILE_STREAM
