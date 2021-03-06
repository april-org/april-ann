
//BIND_HEADER_H
#include "geom_param.h"
#include "bind_image.h"
//BIND_END

//BIND_HEADER_C
using namespace Basics;
using namespace Imaging;
//BIND_END


//BIND_FUNCTION ocr.off_line.param.geom
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
    
  ImageFloat *img;
  const char *param_list;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_OPTIONAL_PARAMETER(2, string, param_list, "siepd");
    
  MatrixFloat *result = OCR::OffLineTextPreprocessing::
    GeomParam::extract(img, param_list);
  LUABIND_RETURN(MatrixFloat, result);
}
//BIND_END

//BIND_FUNCTION ocr.off_line.RLSA
{
  ImageFloat *img;
  int threshold;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_PARAMETER(2, int, threshold);
  OCR::OffLineTextPreprocessing::RLSA(img,threshold);
  // LUABIND_RETURN(ImageFloat,img);
}
//BIND_END

//BIND_FUNCTION ocr.off_line.baselinesRLSA
{
  ImageFloat *img;
  
  float threshold;
  int upper, lower;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, threshold, 0.5f);
  OCR::OffLineTextPreprocessing::baseLinesRLSA(img, &lower, &upper, threshold);
  LUABIND_RETURN(int, lower);
  LUABIND_RETURN(int, upper);
}
//BIND_END
