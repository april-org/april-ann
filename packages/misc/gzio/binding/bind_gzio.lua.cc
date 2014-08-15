/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#include "bind_april_io.h"
#include "stream.h"

using namespace april_io;
//BIND_END

//BIND_HEADER_H
#include "gzfile_stream.h"

using namespace GZIO;
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StreamInterface april_io.stream

//BIND_LUACLASSNAME GZFileStream gzio.stream
//BIND_CPP_CLASS GZFileStream
//BIND_SUBCLASS_OF GZFileStream StreamInterface

//BIND_CONSTRUCTOR GZFileStream
{
  LUABIND_INCREASE_NUM_RETURNS(callFileStreamConstructor<GZFileStream>(L));
}
//BIND_END
