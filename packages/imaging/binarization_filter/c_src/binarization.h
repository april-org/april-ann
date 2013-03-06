#ifndef BINARIZATION_H
#define BINARIZATION_H

#include "utilImageFloat.h"

ImageFloat *binarize_niblack(const ImageFloat *src, int windowRadius, float k, float minThreshold, float maxThreshold);

ImageFloat *binarize_niblack_simple(const ImageFloat *src, int windowRadius,float k);


ImageFloat *binarize_otsus(const ImageFloat *src);

ImageFloat *binarize_threshold(const ImageFloat *src, double threshold);
#endif


