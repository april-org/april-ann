//BIND_HEADER_H
#include "interest_points.h"
#include "utilImageFloat.h"
#include "bind_image.h"
#include "vector.h"
#include "pair.h"
#include <cmath>
#include <cctype>
//BIND_END


//BIND_FUNCTION interest_points.extract_points_from_image
{
  using april_utils::vector;
  using InterestPoints::Point2D;

  LUABIND_CHECK_ARGN(==,1);
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  vector<Point2D> *result=InterestPoints::extract_points_from_image(img);

  lua_createtable(L, result->size(), 0);
  for (unsigned int i=1; i <= result->size(); i++) {
    lua_createtable(L, 2, 0);
    lua_pushnumber(L, (*result)[i-1].first);
    lua_rawseti(L, -2, 1);
    lua_pushnumber(L, (*result)[i-1].second);
    lua_rawseti(L, -2, 2);
    lua_rawseti(L, -2, i);
  }

  LUABIND_RETURN_FROM_STACK(-1); 
  delete result;
}
//BIND_END


