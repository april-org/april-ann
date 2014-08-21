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
#include "zip_package.h"
#include "zipfile_stream.h"

using namespace ZIP;
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_ENUM_CONSTANT zip.flags.NOCASE       ZIP_FL_NOCASE
//BIND_ENUM_CONSTANT zip.flags.NODIR        ZIP_FL_NODIR
//BIND_ENUM_CONSTANT zip.flags.COMPRESSED   ZIP_FL_COMPRESSED
//BIND_ENUM_CONSTANT zip.flags.UNCHANGED    ZIP_FL_UNCHANGED

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StreamInterface aprilio.stream

//BIND_LUACLASSNAME ZIPFileStream zip.stream
//BIND_CPP_CLASS ZIPFileStream
//BIND_SUBCLASS_OF ZIPFileStream StreamInterface

//BIND_CONSTRUCTOR ZIPFileStream
{
  LUABIND_ERROR("Use open method of a zip.package instance");
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ArchivePackage aprilio.package

//BIND_LUACLASSNAME ZIPPackage zip.package
//BIND_CPP_CLASS ZIPPackage
//BIND_SUBCLASS_OF ZIPPackage ArchivePackage

//BIND_CONSTRUCTOR ZIPPackage
{
  LUABIND_INCREASE_NUM_RETURNS(callArchivePackageConstructor<ZIPPackage>(L));
}
//BIND_END
