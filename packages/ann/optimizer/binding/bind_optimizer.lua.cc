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
#include "bind_matrix.h"
//BIND_END

//BIND_HEADER_H
#include "util_rprop.h"
#include "util_regularization.h"
using namespace ANN::Optimizer;
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME UtilRProp ann.optimizer.utils.rprop
//BIND_CPP_CLASS    UtilRProp

//BIND_CONSTRUCTOR UtilRProp
{
  LUABIND_ERROR("Static class, not instantiable");
}
//BIND_END

//BIND_CLASS_METHOD UtilRProp step
{
  LUABIND_CHECK_ARGN(==,5);
  MatrixFloat *steps, *old_sign, *sign;
  float eta_minus, eta_plus;
  LUABIND_GET_PARAMETER(1, MatrixFloat, steps);
  LUABIND_GET_PARAMETER(2, MatrixFloat, old_sign);
  LUABIND_GET_PARAMETER(3, MatrixFloat, sign);
  LUABIND_GET_PARAMETER(4, float, eta_minus);
  LUABIND_GET_PARAMETER(5, float, eta_plus);
  UtilRProp::step(steps, old_sign, sign, eta_minus, eta_plus);
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME UtilRegularization ann.optimizer.utils.regularization
//BIND_CPP_CLASS    UtilRegularization

//BIND_CONSTRUCTOR UtilRegularization
{
  LUABIND_ERROR("Static class, not instantiable");
}
//BIND_END

//BIND_CLASS_METHOD UtilRegularization L1_norm_map
{
  LUABIND_CHECK_ARGN(>=,3);
  MatrixFloat *destw, *w;
  float value;
  LUABIND_GET_PARAMETER(1, MatrixFloat, destw);
  LUABIND_GET_PARAMETER(2, float,       value);
  LUABIND_GET_PARAMETER(3, MatrixFloat, w);
  UtilRegularization::L1NormMap(destw, value, w);
}
//BIND_END
