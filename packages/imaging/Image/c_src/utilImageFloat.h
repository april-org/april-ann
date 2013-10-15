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
#ifndef UTILIMAGEFLOAT
#define UTILIMAGEFLOAT

#include "utilMatrixFloat.h"
#include "matrix.h"
#include "image.h"
#include "floatrgb.h"

typedef Image<float> ImageFloat;
typedef Image<FloatRGB> ImageFloatRGB;

ImageFloat *RGB_to_grayscale(ImageFloatRGB *src);
ImageFloatRGB *grayscale_to_RGB(ImageFloat *src);

#endif // UTILIMAGEFLOAT
