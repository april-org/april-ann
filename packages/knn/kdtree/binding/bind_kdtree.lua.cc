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
  basics::MatrixFloat *m;
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
  LUABIND_RETURN(KDTreeFloat, obj);
}
//BIND_END

//BIND_METHOD KDTreeFloat searchNN
{
  basics::MatrixFloat *point;
  double distance;
  LUABIND_GET_PARAMETER(1,MatrixFloat,point);
  int idx = obj->searchNN(point,distance,0);
  LUABIND_RETURN(int, idx+1);
  LUABIND_RETURN(double, distance);
}
//BIND_END

//BIND_METHOD KDTreeFloat searchKNN
{
  basics::MatrixFloat *point;
  int K;
  LUABIND_GET_PARAMETER(1,int,K);
  LUABIND_GET_PARAMETER(2,MatrixFloat,point);
  april_utils::vector<int> indices;
  april_utils::vector<double> distances;
  obj->searchKNN(K,point,indices,distances,0);
  april_assert(indices.size() == distances.size());
  lua_newtable(L);
  // stack: newtable
  for (size_t i=0; i<indices.size(); ++i) {
    lua_newtable(L);
    // stack: newtable newtable
    lua_pushnumber(L,indices[i]+1);
    // stack: newtable newtable indices[i]
    lua_rawseti(L,-2,1);
    // stack: newtable newtable
    lua_pushnumber(L,distances[i]);
    // stack: newtable newtable distances[i]
    lua_rawseti(L,-2,2);
    // stack: newtable newtable
    lua_rawseti(L,-2,i+1);
    // stack: newtable
  }
  // the table is on the stack top
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD KDTreeFloat get_point_matrix
{
  int index, row;
  LUABIND_GET_PARAMETER(1, int, index);
  basics::MatrixFloat *m = obj->getMatrixAndRow(index-1,row);
  int coords[2] = { row, 0 };
  int sizes[2]  = { 1, obj->getDimSize() };
  LUABIND_RETURN(MatrixFloat, new basics::MatrixFloat(m, coords, sizes, false));
  LUABIND_RETURN(MatrixFloat, m);
}
//BIND_END

//BIND_METHOD KDTreeFloat size
{
  LUABIND_RETURN(int,obj->size());
}
//BIND_END

//BIND_METHOD KDTreeFloat stats
{
  int number_of_processed_points = obj->getNumProcessedPoints();
  LUABIND_RETURN(int, number_of_processed_points);
}
//BIND_END

