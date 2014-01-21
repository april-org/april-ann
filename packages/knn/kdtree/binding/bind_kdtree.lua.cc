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
//BIND_HEADER_C
#include "bind_matrix.h"
#include "bind_mtrand.h"
#include "matrixFloat.h"
//BIND_END

//BIND_HEADER_H
#include "kdtree.h"

using namespace KNN;
//BIND_END

/////////////////////////////////////////////////////
//                  KDTree                         //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME KDTreeFloat knn.kdtree 
//BIND_CPP_CLASS    KDTreeFloat

//BIND_CONSTRUCTOR KDTreeFloat
{
  LUABIND_CHECK_ARGN(==,2);
  int D;
  MTRand *random;
  LUABIND_GET_PARAMETER(1,int,D);
  LUABIND_GET_PARAMETER(2,MTRand,random);
  obj = new KDTree<float>(D,random);
  LUABIND_RETURN(KDTreeFloat, obj);
}
//BIND_END

//BIND_METHOD KDTreeFloat push
{
  LUABIND_CHECK_ARGN(==,1);
  MatrixFloat *m;
  LUABIND_GET_PARAMETER(1,MatrixFloat,m);
  obj->pushMatrix(m);
  LUABIND_RETURN(KDTreeFloat, obj);
}
//BIND_END

//BIND_METHOD KDTreeFloat build
{
  obj->build();
  LUABIND_RETURN(KDTreeFloat, obj);
}
//BIND_END

//BIND_METHOD KDTreeFloat print
{
  obj->print();
}
//BIND_END

//BIND_METHOD KDTreeFloat search
{
}
//BIND_END
