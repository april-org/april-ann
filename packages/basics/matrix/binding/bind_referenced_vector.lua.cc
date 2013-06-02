/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Francisco Zamora-Martinez
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
//BIND_HEADER_H
#include "referenced_vector.h"
#include "matrixFloat.h"
#include "utilLua.h"
#include "bind_matrix.h"
using april_utils::ReferencedVectorFloat;
using april_utils::ReferencedVectorUint;
//BIND_END


//BIND_LUACLASSNAME ReferencedVectorFloat util.vector_float
//BIND_CPP_CLASS    ReferencedVectorFloat

//BIND_CONSTRUCTOR ReferencedVectorFloat
{
  int initial_size;
  LUABIND_CHECK_ARGN(<=, 1);
  LUABIND_GET_OPTIONAL_PARAMETER(1,int,initial_size,1024);
  ReferencedVectorFloat *obj = new ReferencedVectorFloat();
  obj->reserve(initial_size);
  LUABIND_RETURN(ReferencedVectorFloat, obj);
}
//BIND_END

//BIND_METHOD ReferencedVectorFloat get_size
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(int, (int)obj->size());
}
//BIND_END

//BIND_METHOD ReferencedVectorFloat push_back
{
  LUABIND_CHECK_ARGN(==, 1);
  float value;
  LUABIND_GET_PARAMETER(1, float, value);
  obj->push_back(value);
}
//BIND_END

//BIND_METHOD ReferencedVectorFloat toMatrix
{
  // LUABIND_CHECK_ARGN(<=, 1);
  // bool reuse_vector;
  // LUABIND_GET_OPTIONAL_PARAMETER(1,bool,reuse_vector,true);
  int dim = (int)obj->size();
  MatrixFloat *mat;
  // if (reuse_vector) {
  // float *internal_data = obj->release_internal_vector();
  // mat = new MatrixFloat(1,&dim,internal_data);
  // } else {
  mat = new MatrixFloat(1,&dim);
  MatrixFloat::iterator it(mat->begin());
  for (int i=0; i<dim;++i, ++it) {
    *it = (*obj)[i];
  }
  // }
  LUABIND_RETURN(MatrixFloat,mat);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ReferencedVectorUint util.vector_uint
//BIND_CPP_CLASS    ReferencedVectorUint

//BIND_CONSTRUCTOR ReferencedVectorUint
{
  int initial_size;
  LUABIND_CHECK_ARGN(<=, 1);
  LUABIND_GET_OPTIONAL_PARAMETER(1,int,initial_size,1024);
  ReferencedVectorUint *obj = new ReferencedVectorUint();
  obj->reserve(initial_size);
  LUABIND_RETURN(ReferencedVectorUint, obj);
}
//BIND_END

//BIND_METHOD ReferencedVectorUint get_size
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(int, (int)obj->size());
}
//BIND_END

//BIND_METHOD ReferencedVectorUint push_back
{
  LUABIND_CHECK_ARGN(==, 1);
  double value;
  LUABIND_GET_PARAMETER(1, double, value);
  obj->push_back(value);
}
//BIND_END

//BIND_METHOD ReferencedVectorUint toMatrix
{
  // este metodo no hace falta, lo pongo para testear
  LUABIND_CHECK_ARGN(==, 0);
  int dim = (int)obj->size();
  MatrixFloat *mat = new MatrixFloat(1,&dim);
  MatrixFloat::iterator it(mat->begin());
  for (int i=0; i<dim;++i, ++it) {
    uint32_t value = (*obj)[i];
    float aux = value;
    uint32_t test = aux;
    if (value != test)
      LUABIND_FERROR2("vector_uint toMatrix error: %f cannot be converted to uint %u\n",
		      aux,test);
    *it = aux;
  }
  LUABIND_RETURN(MatrixFloat,mat);
}
//BIND_END

