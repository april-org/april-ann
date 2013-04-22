/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco
 * Zamora-Martinez, Joan Pastor-Pellicer
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

#ifndef INTEREST_POINTS_H
#define INTEREST_POINTS_H

#include "image.h"
#include "utilImageFloat.h"
#include "pair.h"
#include "vector.h"

namespace InterestPoints
{
  typedef april_utils::pair<float, float> Point2D;

  /**
   * @brief Given an Image returns a vector with the local maxima and local minima of the given image.
   *
   * @param[in] pimg Pointer to the image
   * @param[in] threshold_white More than this value is considered black
   * @param[in] threshold_black Less than this value is set to white
   * @param[in] local_context The number of pixels in stroke that is used to compute the local maxima/minima
   * @param[in] duplicate_interval The minimum distance of locals within the same stroke
   */
  april_utils::vector<Point2D>* extract_points_from_image(ImageFloat *pimg, float threshold_white = 0.6, float threshold_black = 0.4, int local_context = 6, int duplicate_interval = 3);
}
#endif
