#ifndef INTEREST_POINTS_H
#define INTEREST_POINTS_H

#include "image.h"
#include "utilImageFloat.h"
#include "pair.h"
#include "vector.h"

namespace InterestPoints
{
  typedef april_utils::pair<float, float> Point2D;

  april_utils::vector<Point2D>* extract_points_from_image(ImageFloat *pimg);
}
#endif
