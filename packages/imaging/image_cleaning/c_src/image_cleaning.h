/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Joan Pastor- Pellicer, Salvador España-Boquera, Francisco
 * Zamora-Martinez
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
#ifndef IMAGE_CLEANING_H
#define IMAGE_CLEANING_H

#include <string.h>
#include "referenced.h"
#include "datasetFloat.h"
#include "utilImageFloat.h"
#include "matrix.h"
/**
   Class that contains the counters for calculate the histogram of a given image

**/
//// Given a pixel value, returns the index of histogram
inline int getIndex(float value, int gray_levels) {
  if (value >= 1) return gray_levels - 1;
  return (int) floor(value*gray_levels);
}

class ImageHistogram : public Referenced {
public:

  int gray_levels;
  int *integral_histogram;
  int width, height;


  /// Creator recieves and image
  ImageHistogram(ImageFloat *img, int levels) :
    Referenced(),
    gray_levels(levels){
    this->width  = img->width();
    this->height = img->height();
    integral_histogram = new int[width*height*levels];
    memset(integral_histogram, 0, width*height*levels);
    computeIntegralHistogram(img);
  }

  // Copy Constructor
  ImageHistogram(const ImageHistogram &other);

  /// Destructor
  ~ImageHistogram(){
    delete []integral_histogram;
  }; 
  //Clone
  ImageHistogram *clone() {
    return new ImageHistogram(*this);
  }

  //// Return the total gray levels
  int grayLevels() {
    return gray_levels;
  }

  /// Given a radius gets for each pixel the histogram of these window
  // centered pixel
  Matrix<float> * generateWindowHistogram(int radius);
  //// Return a new copy of the integral matrix
  Matrix<float> * getIntegralHistogram();

  /// Compute all the image Histogram
  Matrix<float> * getImageHistogram();

  Matrix<float> * getWindowHistogram(int x1, int y1, int x2, int y2);
        
  /// Returns a matrix of size Height*Levels
  Matrix<float> *getVerticalHistogram(int radius = 0);
  /// Returns a matrix of size width*levels
  Matrix<float> *getHorizontalHistogram(int radius = 0);

  ///Light and slow image histogram. It's used for computing the histogram of an image on a traditional way (without computing the integral interval)
  static Matrix<float> * getHistogram(const ImageFloat *img, int gray_levels);
  /*{
    int width = img->width;
    int height = img->height;

    int dims[1];

    dims[0] = gray_levels;

    Matrix<float> *matrix = new Matrix<float>(1,dims, 0.0);
    int total = height*width;

    for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
    int h = getIndex((*img)(i,j), gray_levels);
    (*matrix)(h) += 1;      
    }

    }
    //Normalize the histogram
    for(int h = 0; h < gray_levels; ++h) {
    (*matrix)(h) /= total;
    }

    return matrix;
    }*/
protected:
  /// Accessor to the integral_histogram matrix
  inline int hist(int x,int y, int h) const { 
    return integral_histogram[x*width*gray_levels + y*gray_levels+h];
  }
  inline int & hist(int x, int y, int h) {
    return integral_histogram[x*width*gray_levels + y*gray_levels+h];
  }
  //// Takes an Image and Fill the integral matrix
  void computeIntegralHistogram(ImageFloat *img);

  // ImageHistogram* clone();
};

#endif
