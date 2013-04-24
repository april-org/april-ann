/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
class ImageHistogram : public Referenced {
    public:

        int gray_levels;
        int *integral_histogram;
        int width, height;

        //// Creator recieves and image
        ImageHistogram(ImageFloat *img, int levels) :
            gray_levels(levels){
                this->width  = img->width;
                this->height = img->height;
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

        //// Given a radius gets for each pixel the histogram of these window
        // centered pixel
        Matrix<float> * generateWindowHistogram(int radius);
        //// Return a new copy of the integral matrix
        Matrix<float> * getIntegralHistogram();

    protected:
        /// Accessor to the integral_histogram matrix
        inline int hist(int x,int y, int h) const { 
            return integral_histogram[x*width*gray_levels + y*gray_levels+h];
        }
        inline int & hist(int x, int y, int h) {
            return integral_histogram[x*width*gray_levels + y*gray_levels+h];
        }

        //// Given a pixel value, returns the index of histogram
        inline int getIndex(float value) {
            if (value >= 1) return gray_levels - 1;
            return (int) floor(value*gray_levels);
        }
        //// Takes an Image and Fill the integral matrix
        void computeIntegralHistogram(ImageFloat *img);

        // ImageHistogram* clone();
};

#endif
