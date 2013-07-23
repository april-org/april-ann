//BIND_HEADER_C
#include "bind_median_filter.h"
#include "utilMatrixFloat.h"
// #include "bind_image_RGB.h"
#include "bind_image.h"
//BIND_END

//BIND_HEADER_H
#include "bind_image.h"
#include "median_filter.h"
#include "utilMatrixFloat.h"
// #include "utilImageFloat.h"
// #include "bind_matrix.h"
// #include "bind_affine_transform.h"
// #include <cmath>
//BIND_END

//BIND_LUACLASSNAME ImageFloat Image

//BIND_METHOD ImageFloat median_filter
// recibe el radio
{
  int radio;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, int, radio);
  if (radio < 1)
    LUABIND_ERROR("median filter, radio must be > 0");
  LUABIND_RETURN(ImageFloat, medianFilter(obj,radio));
}
//BIND_END

