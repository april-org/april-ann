/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Jorge Gorbe Moya, Joan Pastor Pellicer
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
#ifndef BINARIZATION_H
#define BINARIZATION_H

#include "utilImageFloat.h"

//// Computes Niblack normalization
ImageFloat *binarize_niblack(const ImageFloat *src, int windowRadius, float k, 
                            float minThreshold, float maxThreshold);

//// Another Niblack implementation
ImageFloat *binarize_niblack_simple(const ImageFloat *src,
                                   int windowRadius, float k);

//// Otsu's Binarization
ImageFloat *binarize_otsus(const ImageFloat *src);

/// Simple Image Thresholding
ImageFloat *binarize_threshold(const ImageFloat *src, double threshold);
#endif


