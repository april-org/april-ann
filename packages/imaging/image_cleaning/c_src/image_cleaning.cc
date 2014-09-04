/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
-- * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#include <cmath>
#include "error_print.h"
#include "maxmin.h"
#include "image_cleaning.h"

using AprilMath::MatrixExt::Operations::matZeros;
using namespace Basics;

namespace Imaging {

  MatrixFloat *ImageHistogram::getHistogram(const ImageFloat *img, int gray_levels) {

    int width = img->width();
    int height = img->height();

    int dims[1];
  
    dims[0] = gray_levels;

    MatrixFloat *matrix = new MatrixFloat(1,dims);
    matZeros(matrix);
    int total = height*width;

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int h = getIndex((*img)(j,i), gray_levels);
        (*matrix)(h) += 1;      
      }
    
    }
    //Normalize the histogram
    for(int h = 0; h < gray_levels; ++h) {
      (*matrix)(h) /= total;
    }
  
    return matrix;

  }
  ImageHistogram::ImageHistogram(const ImageHistogram &other) : Referenced() {

    width  = other.width;
    height = other.height;
    gray_levels = other.gray_levels;

    memcpy(other.integral_histogram, integral_histogram, width*height*gray_levels*sizeof(int));

  }
  void ImageHistogram::computeIntegralHistogram(ImageFloat *img) {

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {

        float value = (*img)(j,i);
        // Copy the last histogram
        // First point, nothing to copy
        if (i == 0 && j == 0);
        // First row
        else if (i == 0) {
          for (int h = 0; h < gray_levels; ++h)
            hist(i,j,h) = hist(i, j - 1, h);
        }
        // First column
        else if (j == 0) {
          for (int h = 0; h < gray_levels; ++h)
            hist(i, j, h) = hist(i - 1, j, h);
        }
        // else
        else {
          for (int h = 0; h < gray_levels; ++h)
            hist(i, j, h) = hist(i-1,j,h) + hist(i,j-1,h) - hist(i-1,j-1,h);
        }
        int h = getIndex(value, gray_levels);

        hist(i,j,h) = hist(i,j,h) + 1;
        // First row
      }
    }
  } 

  MatrixFloat * ImageHistogram::generateWindowHistogram(int radius) {

    int dims[3];
    dims[0] = height;
    dims[1] = width;
    dims[2] = gray_levels;

    using AprilUtils::max;
    using AprilUtils::min;

    MatrixFloat *matrix = new MatrixFloat(3,dims);
    april_assert(width > (2*radius+1) && height > (2*radius+1) && "The window is bigger than the image limits");
    matZeros(matrix);

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        // Top-Left Corner
        int top  = max(0, i - radius);
        int left = max(0, j - radius);
        int bottom = min(height - 1 , i + radius);
        int right = min(width - 1, j + radius);
        int window =  (bottom - top)*(right - left); 

        int top_left, top_right, bottom_left, bottom_right;

        // Normalize by size
        int size = (bottom - top + 1)*(right-left + 1);

        for(int h = 0; h < gray_levels; ++h) {

          bottom_right = hist(bottom, right, h);
          top_left = top_right = bottom_left = 0;
          if (top != 0 && left != 0) {
            top_left = hist(top-1, left-1, h);
            bottom_left = hist(bottom, left-1, h);
            top_right = hist(top-1, right, h);
          }
          else if (top == 0 && left == 0);


          else if (top == 0) {
            bottom_left = hist(bottom, left-1, h);
          }
          else if(left == 0) {
            top_right = hist(top-1, right, h);
          }
          int value =  bottom_right + top_left - bottom_left - top_right;
          (*matrix)(i, j, h) = (float)value/size;

        }
      }
    }
    return matrix;
  }

  MatrixFloat *ImageHistogram::getWindowHistogram(int x1, int y1, int x2, int y2){
    int top    = x1;
    int bottom = x2;
    int left   = y1;
    int right  = y2;

    // The returned matrix has the gray level histogram
    int dims[1];
    dims[0] = this->gray_levels;
  
    MatrixFloat *matrix = new MatrixFloat(1, dims);
    matZeros(matrix);

    // Normalize by size
    int size = (bottom - top + 1)*(right-left + 1);
    int top_left, top_right, bottom_left, bottom_right;
    for(int h = 0; h < gray_levels; ++h) {

      bottom_right = hist(bottom, right, h);
      top_left = top_right = bottom_left = 0;

      if (top != 0 && left != 0) {
        top_left = hist(top-1, left-1, h);
        bottom_left = hist(bottom, left-1, h);
        top_right = hist(top-1, right, h);
      }
      else if (top == 0 && left == 0);
      else if (top == 0) {
        bottom_left = hist(bottom, left-1, h);
      }
      else if(left == 0) {
        top_right = hist(top-1, right, h);
      }

      int value =  bottom_right + top_left - bottom_left - top_right;
      (*matrix)(h) = (float)value/size;

    }
    return matrix;

  }

  MatrixFloat * ImageHistogram::getImageHistogram() {

    return getWindowHistogram(0,0, height - 1, width - 1);
  }

  MatrixFloat * ImageHistogram::getHorizontalHistogram(int radius) {
  
    using AprilUtils::max;
    using AprilUtils::min;
    int dims[2];
    dims[0] = this->height;
    dims[1] = this->gray_levels;

    MatrixFloat *vHist = new MatrixFloat(2,dims);

    for (int i = 0; i < this->height; ++i) {

      //FIXME: Memory allocation on each line
    
      MatrixFloat *m = getWindowHistogram(max(i - radius,0) ,0, min(height-1, i + radius), width-1);
      //TODO: Copy on efficient way
      for (int h = 0; h < this->gray_levels; ++h)
        (*vHist)(i,h) = (*m)(h);
      delete m;
    }

    return vHist;
  }

  MatrixFloat * ImageHistogram::getVerticalHistogram(int radius) {

    using AprilUtils::max;
    using AprilUtils::min;
    int dims[2];
    dims[0] = this->width;
    dims[1] = this->gray_levels;

    MatrixFloat *vHist = new MatrixFloat(2,dims);

    for (int i = 0; i < this->width; ++i) {

      //FIXME: Memory allocation on each line
      MatrixFloat *m = getWindowHistogram(0,max(0, i-radius), height-1, min(width-1,i+radius));
      //TODO: Copy on efficient way
      for (int h = 0; h < this->gray_levels; ++h)
        (*vHist)(i,h) = (*m)(h);
      delete m;
    }

    return vHist;
  }

  MatrixFloat * ImageHistogram::getIntegralHistogram(){
    using AprilUtils::max;
    using AprilUtils::min;
    int dims[3];
    dims[0] = height;
    dims[1] = width;
    dims[2] = gray_levels;
    MatrixFloat *matrix = new MatrixFloat(3, dims);

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        for(int h = 0; h < gray_levels; ++h) {
          (*matrix)(i, j, h) = (float)hist(i, j, h); 
        }
      }
    }
    return matrix;
  }

} // namespace Imaging
