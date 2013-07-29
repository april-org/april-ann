/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013 Francisco Zamora-Martinez
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
//BIND_HEADER_C
#include "bind_matrix.h"
//BIND_END

//BIND_HEADER_H
#include "matlab.h"
//BIND_END

//BIND_FUNCTION matlab.read
{
  LUABIND_CHECK_ARGN(==,1);
  const char *path;
  LUABIND_GET_PARAMETER(1, string, path);
  MatFileReader reader(path);
  char name[MAX_NAME_SIZE+1];
  MatrixFloat *m;
  while((m=reader.readNextMatrix(name))!=0)
    LUABIND_RETURN(MatrixFloat, m);
}
//BIND_END
