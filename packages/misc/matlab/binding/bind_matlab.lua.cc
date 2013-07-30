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

int elements_iterator_function(lua_State *L) {
  MatFileReader *obj = lua_toMatFileReader(L,1);
  MatDataElement *element = obj->getNextDataElement();
  if (element == 0) {
    lua_pushnil(L);
    return 1;
  }
  lua_pushMatDataElement(L, element);
  return 1;
}

//BIND_END

//BIND_HEADER_H
#include "matlab.h"
typedef MatFileReader::DataElement MatDataElement;
typedef MatFileReader::CellArrayDataElement MatCellArrayDataElement;
//BIND_END

//BIND_LUACLASSNAME MatFileReader matlab.reader
//BIND_CPP_CLASS    MatFileReader

//BIND_LUACLASSNAME MatDataElement matlab.element
//BIND_CPP_CLASS    MatDataElement

//BIND_LUACLASSNAME MatCellArrayDataElement matlab.cell_array
//BIND_CPP_CLASS    MatCellArrayDataElement

//BIND_ENUM_CONSTANT matlab.types.matrix MatFileReader::MATRIX
//BIND_ENUM_CONSTANT matlab.types.int8   MatFileReader::INT8
//BIND_ENUM_CONSTANT matlab.types.int16  MatFileReader::INT16
//BIND_ENUM_CONSTANT matlab.types.int32  MatFileReader::INT32
//BIND_ENUM_CONSTANT matlab.types.int64  MatFileReader::INT64
//BIND_ENUM_CONSTANT matlab.types.uint8  MatFileReader::UINT8
//BIND_ENUM_CONSTANT matlab.types.uint16 MatFileReader::UINT16
//BIND_ENUM_CONSTANT matlab.types.uint32 MatFileReader::UINT32
//BIND_ENUM_CONSTANT matlab.types.uint64 MatFileReader::UINT64
//BIND_ENUM_CONSTANT matlab.types.single MatFileReader::SINGLE
//BIND_ENUM_CONSTANT matlab.types.double MatFileReader::DOUBLE
//BIND_ENUM_CONSTANT matlab.types.utf8   MatFileReader::UTF8
//BIND_ENUM_CONSTANT matlab.types.utf16  MatFileReader::UTF16
//BIND_ENUM_CONSTANT matlab.types.utf32  MatFileReader::UTF32

//BIND_ENUM_CONSTANT matlab.classes.cell_array MatFileReader::CL_CELL_ARRAY
//BIND_ENUM_CONSTANT matlab.classes.structure  MatFileReader::CL_STRUCTURE
//BIND_ENUM_CONSTANT matlab.classes.object     MatFileReader::CL_OBJECT
//BIND_ENUM_CONSTANT matlab.classes.char       MatFileReader::CL_CHAR
//BIND_ENUM_CONSTANT matlab.classes.sparse     MatFileReader::CL_SPARSE
//BIND_ENUM_CONSTANT matlab.classes.double     MatFileReader::CL_DOUBLE
//BIND_ENUM_CONSTANT matlab.classes.single     MatFileReader::CL_SINGLE
//BIND_ENUM_CONSTANT matlab.classes.int8       MatFileReader::CL_INT8
//BIND_ENUM_CONSTANT matlab.classes.uint8      MatFileReader::CL_UINT8
//BIND_ENUM_CONSTANT matlab.classes.int16      MatFileReader::CL_INT16
//BIND_ENUM_CONSTANT matlab.classes.uint16     MatFileReader::CL_UINT16
//BIND_ENUM_CONSTANT matlab.classes.int32      MatFileReader::CL_INT32
//BIND_ENUM_CONSTANT matlab.classes.uint32     MatFileReader::CL_UINT32

//BIND_CONSTRUCTOR MatFileReader
{
  LUABIND_CHECK_ARGN(==,1);
  const char *path;
  LUABIND_GET_PARAMETER(1, string, path);
  MatFileReader *reader = new MatFileReader(path);
  LUABIND_RETURN(MatFileReader, reader);
}
//BIND_END

//BIND_METHOD MatFileReader reset
{
  obj->reset();
}
//BIND_END

//BIND_METHOD MatFileReader get_next_element
{
  MatDataElement *element = obj->getNextDataElement();
  LUABIND_RETURN(MatDataElement, element);
}
//BIND_END

//BIND_METHOD MatFileReader elements
{
  LUABIND_RETURN(cfunction,elements_iterator_function);
  LUABIND_RETURN(MatFileReader,obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatDataElement
{
  LUABIND_ERROR("abstract class");
}
//BIND_END

//BIND_METHOD MatDataElement get_type
{
  LUABIND_RETURN(uint, obj->getDataType());
}
//BIND_END

//BIND_METHOD MatDataElement get_class
{
  LUABIND_RETURN(uint, obj->getClass());
}
//BIND_END

//BIND_METHOD MatDataElement get_matrix
{
  char name[MAX_NAME_SIZE];
  MatrixFloat *m = obj->getMatrix(name, MAX_NAME_SIZE, false);
  LUABIND_RETURN(MatrixFloat, m);
  LUABIND_RETURN(string, name);
}
//BIND_END

//BIND_METHOD MatDataElement get_cell_array
{
  char name[MAX_NAME_SIZE];
  MatCellArrayDataElement *c = obj->getCellArray(name, MAX_NAME_SIZE);
  LUABIND_RETURN(MatCellArrayDataElement, c);
  LUABIND_RETURN(string, name);
}
//BIND_END

//BIND_METHOD MatDataElement reset
{
  obj->reset();
}
//BIND_END

//BIND_METHOD MatDataElement get_num_bytes
{
  LUABIND_RETURN(uint, obj->getNumberOfBytes());
}
//BIND_END

//BIND_METHOD MatDataElement get_next_subelement
{
  if (obj->getNextSubElement() != 0)
    LUABIND_RETURN(MatDataElement, obj->getNextSubElement());
  else
    LUABIND_RETURN_NIL();
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatCellArrayDataElement
{
  LUABIND_ERROR("abstract class\n");
}
//BIND_END

//BIND_METHOD MatCellArrayDataElement get
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  int *coords = new int[obj->getNumDim()];
  for (int i=0; i<obj->getNumDim(); ++i) {
    LUABIND_GET_PARAMETER(i+1,int,coords[i]);
    if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      coords[i], obj->getDimSize(i));
    }
    coords[i]--;
  }
  MatDataElement *e = obj->getElementAt(coords, argn);
  delete[] coords;
  LUABIND_RETURN(MatDataElement, e);
}
//BIND_END

//BIND_METHOD MatCellArrayDataElement dim
{
  int argn = lua_gettop(L); // number of arguments
  if (argn == 1) {
    int idx;
    LUABIND_GET_PARAMETER(1, int, idx);
    LUABIND_RETURN(int, obj->getDimSize(idx));
  }
  else {
    const int *d = obj->getDimPtr();
    LUABIND_VECTOR_TO_NEW_TABLE(int, d, obj->getNumDim());
    LUABIND_RETURN_FROM_STACK(-1);
  }
}
//BIND_END
