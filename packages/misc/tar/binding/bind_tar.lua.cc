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

using namespace AprilIO;
//BIND_END

//BIND_HEADER_H
#include "tar_package.h"
#include "tarfile_stream.h"

using namespace TAR;
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StreamInterface aprilio.stream

//BIND_LUACLASSNAME TARFileStream tar.stream
//BIND_CPP_CLASS TARFileStream
//BIND_SUBCLASS_OF TARFileStream StreamInterface

//BIND_CONSTRUCTOR TARFileStream
{
  LUABIND_ERROR("Use open method of a tar.package instance");
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ArchivePackage aprilio.package

//BIND_LUACLASSNAME TARPackage tar.package
//BIND_CPP_CLASS TARPackage
//BIND_SUBCLASS_OF TARPackage ArchivePackage

//BIND_CONSTRUCTOR TARPackage
{
  LUABIND_INCREASE_NUM_RETURNS(callArchivePackageConstructor<TARPackage>(L));
}
//BIND_END
