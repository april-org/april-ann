//BIND_HEADER_H
#include "off_line_text_preprocessing.h"
#include "utilImageFloat.h"
#include "bind_image.h"
#include "vector.h"
#include <cmath>
#include <cctype>
#include "geometry.h"
//BIND_END

//BIND_FUNCTION ocr.off_line_text_preprocessing.normalize_image
{
  using april_utils::vector;
  using april_utils::Point2D;

  LUABIND_CHECK_ARGN(==,2);
  ImageFloat *img;
  int dst_height;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_PARAMETER(2, int, dst_height);

  ImageFloat *result = OCR::OffLineTextPreprocessing::normalize_image(img, dst_height);

  LUABIND_RETURN(ImageFloat, result);
}
//BIND_END
//BIND_FUNCTION ocr.off_line_text_preprocessing.normalize_size
{
  using april_utils::vector;
  using april_utils::Point2D;

  LUABIND_CHECK_ARGN(>=,7);
  LUABIND_CHECK_ARGN(<=,9);
  ImageFloat *img;
  float ascender_ratio, descender_ratio;
  int dst_height;
  bool keep_aspect;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_PARAMETER(2, float, ascender_ratio);
  LUABIND_GET_PARAMETER(3, float, descender_ratio);
  LUABIND_CHECK_PARAMETER(4, table);
  LUABIND_CHECK_PARAMETER(5, table);
  LUABIND_CHECK_PARAMETER(6, table);
  LUABIND_CHECK_PARAMETER(7, table);

  LUABIND_GET_OPTIONAL_PARAMETER(8, int, dst_height, -1);

  LUABIND_GET_OPTIONAL_PARAMETER(9, bool, keep_aspect, false);
  int point_vector_length[4];
  vector<Point2D> point_vector[4];
  
  for (int vec=0; vec<4; vec++) {
    LUABIND_TABLE_GETN(vec+4, point_vector_length[vec]);
    point_vector[vec].resize(point_vector_length[vec]);
    for (int i=1; i <= point_vector_length[vec]; ++i) {
      lua_rawgeti(L,vec+4,i); // punto i-esimo, es una tabla, los vectores estan en pila[4-7]
      lua_rawgeti(L,-1,1); // x
      LUABIND_GET_PARAMETER(-1, float, point_vector[vec][i-1].x);
      lua_pop(L,1); // la quitamos de la pila
      lua_rawgeti(L,-1,2); // y
      LUABIND_GET_PARAMETER(-1, float, point_vector[vec][i-1].y);
      lua_pop(L,2); // la quitamos de la pila, tb la tabla
    }
  }

  ImageFloat *result = OCR::OffLineTextPreprocessing::normalize_size(img, ascender_ratio, descender_ratio,
      point_vector[0], point_vector[1], point_vector[2], point_vector[3], dst_height, keep_aspect);

  LUABIND_RETURN(ImageFloat, result);
}
//BIND_END

//BIND_FUNCTION ocr.off_line_text_preprocessing.add_asc_desc
{

    LUABIND_CHECK_ARGN(==,2);
    ImageFloat *img;
    MatrixFloat *mat;

    LUABIND_GET_PARAMETER(1, ImageFloat, img); 
    LUABIND_GET_PARAMETER(2, MatrixFloat, mat); 
    MatrixFloat *result = OCR::OffLineTextPreprocessing::add_asc_desc(img,mat);
    LUABIND_RETURN(MatrixFloat, result);
}
//BIND_END

//BIND_FUNCTION ocr.off_line_text_preprocessing.normalize_from_matrix
{
  using april_utils::vector;
  using april_utils::Point2D;

  LUABIND_CHECK_ARGN(>=,4);
  LUABIND_CHECK_ARGN(<=,6);
  ImageFloat *img;
  MatrixFloat *mat;
  float ascender_ratio, descender_ratio;
  int dst_height;
  bool keep_aspect;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);
  LUABIND_GET_PARAMETER(4, MatrixFloat, mat);
  LUABIND_GET_PARAMETER(2, float, ascender_ratio);
  LUABIND_GET_PARAMETER(3, float, descender_ratio);

  LUABIND_GET_OPTIONAL_PARAMETER(5, int, dst_height, -1);

  LUABIND_GET_OPTIONAL_PARAMETER(6, bool, keep_aspect, false);

  ImageFloat *result = OCR::OffLineTextPreprocessing::normalize_size(img, mat, ascender_ratio, descender_ratio,
      dst_height, keep_aspect);

  LUABIND_RETURN(ImageFloat, result);
}
//BIND_END
