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
#include "bind_matrix_complex_float.h"
#include "bind_matrix_char.h"
#include "bind_matrix_int32.h"
#include "bind_matrix_double.h"

namespace matlab {
  
  int elements_iterator_function(lua_State *L) {
    MatFileReader *obj = lua_toMatFileReader(L,1);
    MatTaggedDataElement *element = obj->getNextDataElement();
    if (element == 0) {
      lua_pushnil(L);
      return 1;
    }
    lua_pushMatTaggedDataElement(L, element);
    return 1;
  }

} // namespace matlab

//BIND_END

//BIND_HEADER_H
#include "matlab.h"

using namespace matlab;

typedef MatFileReader::DataElementInterface MatDataElementInterface;
typedef MatFileReader::TaggedDataElement    MatTaggedDataElement;
typedef MatFileReader::CellArrayDataElement MatCellArrayDataElement;
typedef MatFileReader::StructureDataElement MatStructureDataElement;
//BIND_END

//BIND_LUACLASSNAME MatFileReader matlab.reader
//BIND_CPP_CLASS    MatFileReader

//BIND_LUACLASSNAME MatDataElementInterface matlab.__element_interface__
//BIND_CPP_CLASS    MatDataElementInterface

//BIND_LUACLASSNAME MatTaggedDataElement matlab.tagged_element
//BIND_CPP_CLASS    MatTaggedDataElement
//BIND_SUBCLASS_OF  MatTaggedDataElement MatDataElementInterface

//BIND_LUACLASSNAME MatCellArrayDataElement matlab.cell_array
//BIND_CPP_CLASS    MatCellArrayDataElement
//BIND_SUBCLASS_OF  MatCellArrayDataElement MatDataElementInterface

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
//BIND_ENUM_CONSTANT matlab.classes.int64      MatFileReader::CL_INT64
//BIND_ENUM_CONSTANT matlab.classes.uint64     MatFileReader::CL_UINT64

/////////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatDataElementInterface
{
  LUABIND_ERROR("Abstract class");
}
//BIND_END

//BIND_METHOD MatDataElementInterface get_type
{
  LUABIND_RETURN(uint, obj->getDataType());
}
//BIND_END

//BIND_METHOD MatDataElementInterface get_class
{
  LUABIND_RETURN(uint, obj->getClass());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

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
  MatTaggedDataElement *element = obj->getNextDataElement();
  LUABIND_RETURN(MatTaggedDataElement, element);
}
//BIND_END

//BIND_METHOD MatFileReader elements
{
  LUABIND_RETURN(cfunction,elements_iterator_function);
  LUABIND_RETURN(MatFileReader,obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatTaggedDataElement
{
  LUABIND_ERROR("abstract class");
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_matrix
{
  bool col_major;
  LUABIND_GET_OPTIONAL_PARAMETER(1, bool, col_major, false);
  char name[MAX_NAME_SIZE];
  basics::MatrixFloat *m = obj->getMatrix(name, MAX_NAME_SIZE, col_major);
  if (m != 0) {
    LUABIND_RETURN(MatrixFloat, m);
    LUABIND_RETURN(string, name);
  }
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_matrix_complex
{
  bool col_major;
  LUABIND_GET_OPTIONAL_PARAMETER(1, bool, col_major, false);
  char name[MAX_NAME_SIZE];
  basics::MatrixComplexF *m = obj->getMatrixComplexF(name, MAX_NAME_SIZE, col_major);
  LUABIND_RETURN(MatrixComplexF, m);
  LUABIND_RETURN(string, name);
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_matrix_double
{
  char name[MAX_NAME_SIZE];
  basics::MatrixDouble *m = obj->getMatrixDouble(name, MAX_NAME_SIZE);
  if (m != 0) {
    LUABIND_RETURN(MatrixDouble, m);
    LUABIND_RETURN(string, name);
  }
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_matrix_char
{
  char name[MAX_NAME_SIZE];
  basics::MatrixChar *m = obj->getMatrixChar(name, MAX_NAME_SIZE);
  LUABIND_RETURN(MatrixChar, m);
  LUABIND_RETURN(string, name);
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_matrix_int32
{
  char name[MAX_NAME_SIZE];
  basics::MatrixInt32 *m = obj->getMatrixInt32(name, MAX_NAME_SIZE);
  if (m != 0) {
    LUABIND_RETURN(MatrixInt32, m);
    LUABIND_RETURN(string, name);
  }
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_cell_array
{
  char name[MAX_NAME_SIZE];
  MatCellArrayDataElement *c = obj->getCellArray(name, MAX_NAME_SIZE);
  LUABIND_RETURN(MatCellArrayDataElement, c);
  LUABIND_RETURN(string, name);
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_structure
{
  char name[MAX_NAME_SIZE];
  MatStructureDataElement *s = obj->getStructure(name, MAX_NAME_SIZE);
  lua_createtable(L, 0, s->size());
  for (MatStructureDataElement::HashType::iterator it=s->begin();
       it != s->end(); ++it) {
    const char *ename = it->first.c_str();
    MatTaggedDataElement *e = it->second;
    lua_pushstring(L, ename);
    lua_pushMatTaggedDataElement(L, e);
    lua_rawset(L, -3);
  }
  LUABIND_RETURN_FROM_STACK(-1);
  LUABIND_RETURN(string, name);
  delete s;
}
//BIND_END

//BIND_METHOD MatTaggedDataElement reset
{
  obj->reset();
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_num_bytes
{
  LUABIND_RETURN(uint, obj->getNumberOfBytes());
}
//BIND_END

//BIND_METHOD MatTaggedDataElement get_next_subelement
{
  if (obj->getNextSubElement() != 0)
    LUABIND_RETURN(MatTaggedDataElement, obj->getNextSubElement());
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

//BIND_METHOD MatCellArrayDataElement compute_coords
{
  LUABIND_CHECK_ARGN(==,1);
  int raw_idx;
  LUABIND_GET_PARAMETER(1, int, raw_idx);
  if (raw_idx < 0 || raw_idx >= obj->getSize())
    LUABIND_FERROR3("Incorrect raw index, found %d, expected betwen [%d,%d]",
		    raw_idx, 0, obj->getSize());
  int *coords = new int[obj->getNumDim()];
  obj->computeCoords(raw_idx, coords);
  for (int i=0; i<obj->getNumDim(); ++i) coords[i]++;
  LUABIND_VECTOR_TO_NEW_TABLE(int, coords, obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
  delete[] coords;
}
//BIND_END

//BIND_METHOD MatCellArrayDataElement raw_get
{
  LUABIND_CHECK_ARGN(==,1);
  int raw_idx;
  LUABIND_GET_PARAMETER(1, int, raw_idx);
  if (raw_idx < 0 || raw_idx >= obj->getSize())
    LUABIND_FERROR3("Incorrect raw index, found %d, expected betwen [%d,%d]",
		    raw_idx, 0, obj->getSize());
  LUABIND_RETURN(MatTaggedDataElement, obj->getElementAt(raw_idx));
}
//BIND_END

//BIND_METHOD MatCellArrayDataElement size
{
  LUABIND_RETURN(int, obj->getSize());
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
  MatTaggedDataElement *e = obj->getElementAt(coords, argn);
  delete[] coords;
  LUABIND_RETURN(MatTaggedDataElement, e);
}
//BIND_END

//BIND_METHOD MatCellArrayDataElement dim
{
  int argn = lua_gettop(L); // number of arguments
  if (argn == 1) {
    int idx;
    LUABIND_GET_PARAMETER(1, int, idx);
    if (idx < 1 || idx > obj->getNumDim())
      LUABIND_FERROR3("Incorrect dimension number, found %d, expected "
		      "between [%d,%d]", idx, 1, obj->getNumDim());
    LUABIND_RETURN(int, obj->getDimSize(idx-1));
  }
  else {
    const int *d = obj->getDimPtr();
    LUABIND_VECTOR_TO_NEW_TABLE(int, d, obj->getNumDim());
    LUABIND_RETURN_FROM_STACK(-1);
  }
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatStructureDataElement
{
  LUABIND_ERROR("abstract class\n");
}
//BIND_END
