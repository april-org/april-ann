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
#include "error_print.h"
#include "image_cleaning.h"
#include "maxmin.h"
#include <cmath>

ImageHistogram::ImageHistogram(const ImageHistogram &other) {

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
            int h = getIndex(value);

            hist(i,j,h) = hist(i,j,h) + 1;
            // First row
        }
    }
} 

Matrix<float> * ImageHistogram::generateWindowHistogram(int radius) {

    int dims[3];
    dims[0] = height;
    dims[1] = width;
    dims[2] = gray_levels;

    using april_utils::max;
    using april_utils::min;

    Matrix<float> *matrix = new Matrix<float>(3,dims, 0.0f);
    assert(width > (2*radius+1) && height > (2*radius+1) && "The window is bigger than the image limits");

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

Matrix<float> * ImageHistogram::getIntegralHistogram(){
    int dims[3];
    dims[0] = height;
    dims[1] = width;
    dims[2] = gray_levels;
    Matrix<float> *matrix = new Matrix<float>(3, dims, 0.0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for(int h = 0; h < gray_levels; ++h) {
                (*matrix)(i, j, h) = (float)hist(i, j, h); 
            }
        }
    }
    return matrix;
}

