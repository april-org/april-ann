/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Jorge Gorbe Moya
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
#include "utilImageFloat.h"

namespace imaging {

  ImageFloat *RGB_to_grayscale(ImageFloatRGB *src)
  {
    int dims[2]={src->height(), src->width()};
    basics::MatrixFloat *m = new basics::MatrixFloat(2, dims);
    ImageFloat *result = new ImageFloat(m);

    for (int y=0; y < src->height(); y++) {
      for (int x=0; x < src->width(); x++) {
        FloatRGB rgb = (*src)(x,y);
        (*result)(x,y) = rgb.to_grayscale();
      }
    }

    return result;
  }

  ImageFloatRGB *grayscale_to_RGB(ImageFloat *src)
  {
    int dims[2]={src->height(), src->width()};
    basics::Matrix<FloatRGB> *m = new basics::Matrix<FloatRGB>(2, dims);
    ImageFloatRGB *result = new ImageFloatRGB(m);

    for (int y=0; y < src->height(); y++) {
      for (int x=0; x < src->width(); x++) {
        float val = (*src)(x,y);
        (*result)(x,y) = FloatRGB(val);
      }
    }

    return result;
  }

  template<>
  Image<FloatRGB> *Image<FloatRGB>::convolution5x5(float *k,
                                                   FloatRGB default_color) const
  {
    UNUSED_VARIABLE(k);
    UNUSED_VARIABLE(default_color);
    ERROR_EXIT(256, "Not implemented for RGB images\n");
    return 0;
  }

} // namespace imaging
