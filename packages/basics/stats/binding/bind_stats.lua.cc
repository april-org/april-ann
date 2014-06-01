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
//BIND_END

//BIND_HEADER_H
#include "statistical_distribution.h"
#include "uniform_distribution.h"
#include "normal_distribution.h"

using namespace Stats;

//BIND_END

//BIND_LUACLASSNAME StatisticalDistributionBase stats.dist
//BIND_CPP_CLASS    StatisticalDistributionBase

//BIND_CONSTRUCTOR StatisticalDistributionBase
{
  LUABIND_ERROR("Abstract class");
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase sample
{
  MTRand *rng;
  MatrixFloat *dest;
  LUABIND_GET_PARAMETER(1, MTRand, rng);
  LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
  LUABIND_RETURN(MatrixFloat, obj->sample(rng, dest));
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase logpdf
{
  MatrixFloat *x;
  LUABIND_GET_PARAMETER(1, MatrixFloat, x);
  LUABIND_RETURN(float, obj->logpdf(x).log());
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase logcdf
{
  MatrixFloat *x;
  LUABIND_GET_PARAMETER(1, MatrixFloat, x);
  LUABIND_RETURN(float, obj->logcdf(x).log());
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase clone
{
  LUABIND_RETURN(StatisticalDistributionBase, obj->clone());
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase get_params
{
  LUABIND_RETURN(MatrixFloatSet, obj->getParams());
}
//BIND_END

//BIND_METHOD StatisticalDistributionBase to_lua_string
{
  const char *format;
  bool is_ascii = false;
  LUABIND_GET_OPTIONAL_PARAMETER(1, string, format, "binary");
  if (strcmp(format,"ascii") == 0)
    is_ascii = true;
  else if (strcmp(format,"binary") != 0)
    LUABIND_FERROR1("Incorrect format, expected 'ascii' or 'binary', given '%s'",
                    format);
  char *str = obj->toLuaString(is_ascii);
  LUABIND_RETURN(string, str);
  delete[] str;
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME UniformDistribution stats.dist.uniform
//BIND_CPP_CLASS    UniformDistribution
//BIND_SUBCLASS_OF  UniformDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR UniformDistribution
{
  MatrixFloat *low, *high;
  LUABIND_GET_PARAMETER(1, MatrixFloat, low);
  LUABIND_GET_PARAMETER(2, MatrixFloat, high);
  obj = new UniformDistribution(low, high);
  LUABIND_RETURN(UniformDistribution, obj);
}
//BIND_END

//BIND_METHOD UniformDistribution clone
{
  LUABIND_RETURN(UniformDistribution,
                 static_cast<UniformDistribution*>(obj->clone()));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME GeneralNormalDistribution stats.dist.normal
//BIND_CPP_CLASS    GeneralNormalDistribution
//BIND_SUBCLASS_OF  GeneralNormalDistribution StatisticalDistributionBase

//BIND_CONSTRUCTOR GeneralNormalDistribution
{
  MatrixFloat *mean, *cov;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mean);
  LUABIND_GET_PARAMETER(2, MatrixFloat, cov);
  obj = new GeneralNormalDistribution(mean, cov);
  LUABIND_RETURN(GeneralNormalDistribution, obj);
}
//BIND_END

//BIND_METHOD GeneralNormalDistribution clone
{
  LUABIND_RETURN(GeneralNormalDistribution,
                 static_cast<GeneralNormalDistribution*>(obj->clone()));
}
//BIND_END
