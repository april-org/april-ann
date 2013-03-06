//BIND_HEADER_C
//BIND_END

//BIND_HEADER_H
#include "bind_image.h"
#include "binarization.h"
#include "utilMatrixFloat.h"
//BIND_END

//BIND_LUACLASSNAME ImageFloat Image

//BIND_METHOD ImageFloat binarize_niblack
{
  int radius;
  float k, minThreshold, maxThreshold;
  LUABIND_CHECK_ARGN(==, 4);
  LUABIND_GET_PARAMETER(1, int, radius);
  LUABIND_GET_PARAMETER(2, float, k);
  LUABIND_GET_PARAMETER(3, float, minThreshold);
  LUABIND_GET_PARAMETER(4, float, maxThreshold);
  if (radius < 1)
    LUABIND_ERROR("median filter, radius must be > 0");
  LUABIND_RETURN(ImageFloat, binarize_niblack(obj,radius, k, minThreshold, maxThreshold));
}
//BIND_END

//BIND_METHOD ImageFloat binarize_niblack_simple
{
  int radius;
  float k, minThreshold, maxThreshold;
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_GET_PARAMETER(1, int, radius);
  LUABIND_GET_OPTIONAL_PARAMETER(2,float, k, 0.2);
  if (radius < 1)
    LUABIND_ERROR("median filter, radius must be > 0");
  LUABIND_RETURN(ImageFloat, binarize_niblack_simple(obj,radius, k));
}
//BIND_END

//BIND_METHOD ImageFloat binarize_otsus
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(ImageFloat, binarize_otsus(obj));
}
//BIND_END

//BIND_METHOD ImageFloat binarize_threshold
{
  double threshold;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, double, threshold);
  LUABIND_RETURN(ImageFloat, binarize_threshold(obj, threshold));
}
//BIND_END

