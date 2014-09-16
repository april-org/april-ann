/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#include <cmath>
#include "binarization.h"

namespace Imaging {

  ImageFloat *binarize_niblack(const ImageFloat *src, int windowRadius, float k, float minThreshold, float maxThreshold)
  {
    april_assert(src->width()  > 0 && "Zero-sized image!");
    april_assert(src->height() > 0 && "Zero-sized image!");

    ImageFloat *result       = new ImageFloat(src->width(), src->height());
    ImageFloat *sum          = new ImageFloat(src->width(), src->height());
    ImageFloat *sumOfSquares = new ImageFloat(src->width(), src->height());

    // each pixel in the "sum" image contains the sum of the pixels to the left
    // and above it, same for sumOfSquares

    // init to the upper-left corner pixel
    (*sum)(0,0) = (*src)(0,0);
    (*sumOfSquares)(0,0) = (*src)(0,0) * (*src)(0,0);

    // First row
    for (int x=1; x < src->width(); x++) {
      float current_pixel = (*src)(x,0);
      (*sum)(x,0) = (*sum)(x-1, 0) + current_pixel;
      (*sumOfSquares)(x,0) = (*sumOfSquares)(x-1, 0) + current_pixel * current_pixel;
    }

    // Rest of rows
    for (int y=1; y < src->height(); y++)
      for (int x=0; x < src->width(); x++) {
        float s  = (*sum)(x,y-1);
        float s2 = (*sumOfSquares)(x,y-1);
        if (x>0) {
          s  += (*sum)(x-1,y) - (*sum)(x-1,y-1);
          s2 += (*sumOfSquares)(x-1, y) - (*sumOfSquares)(x-1,y-1);
        }
      
        float current_pixel = (*src)(x,y);
        (*sum)(x,y) = s + current_pixel;
        (*sumOfSquares)(x,y) = s + current_pixel * current_pixel;
      }
 

    // Apply Niblack filter using sum and sumOfSquares for fast mean/std.dev. computation
    int windowSize = 2*windowRadius+1;
    int totalWindowPixels = windowSize*windowSize;
    for (int y=0; y < src->height(); y++)
      for (int x=0; x < src->width(); x++) {
        float val = (*src)(x,y);
        if (val < minThreshold) {
          (*result)(x,y) = 0;
        }
        else if (val > maxThreshold) {
          (*result)(x,y) = 1;
        }
        else {
          int windowUpper, windowLower, windowLeft, windowRight;
        
          windowUpper = (y-windowRadius < 0 ? 0 : y-windowRadius);
          windowLeft  = (x-windowRadius < 0 ? 0 : x-windowRadius);
          windowLower = (y+windowRadius > src->height() - 1 ? src->height()-1 : y+windowRadius);
          windowRight = (x+windowRadius > src->width()  - 1 ? src->width() -1 : x+windowRadius);

          // assume pixels outside the image are white (value = 1)
          int windowPixels = (windowRight-windowLeft+1) * (windowLower-windowUpper+1);
          float s = totalWindowPixels - windowPixels;
          float s2 = s; // 1 squared is 1, too

          s  += (*sum)(windowRight, windowLower);
          s2 += (*sumOfSquares)(windowRight, windowLower);

          if (windowLeft > 0) {
            s  -= (*sum)(windowLeft-1, windowLower);
            s2 -= (*sumOfSquares)(windowLeft-1, windowLower);
          }

          if (windowUpper > 0) {
            s  -= (*sum)(windowRight, windowUpper-1);
            s2 -= (*sumOfSquares)(windowRight, windowUpper-1);
          }
        
          if (windowLeft > 0 && windowUpper > 0) {
            s  += (*sum)(windowLeft-1, windowUpper-1);
            s2 += (*sumOfSquares)(windowLeft-1, windowUpper-1);
          }

          float mean = s/totalWindowPixels;
          float std_dev = sqrt(s2/totalWindowPixels - mean*mean);
          float threshold = mean + k*std_dev;

          (*result)(x,y) = val < threshold ? 0 : 1;
        }
      }
  
    return result;
  }




  ImageFloat *binarize_niblack_simple(const ImageFloat *src, int windowRadius, float k)
  {
    april_assert(src->width()  > 0 && "Zero-sized image!");
    april_assert(src->height() > 0 && "Zero-sized image!");
    // Image Integral Matrix
    ImageFloat *result = new ImageFloat(src->width(), src->height());
    double **M = new double*[src->width()];
    for(int i = 0; i < src->width(); ++i)
      M[i] = new double[src->height()];
    // Square Image Integral
    double **M2 = new double*[src->width()];
    for(int i = 0; i < src->width(); ++i)
      M2[i] = new double[src->height()];

    int env=windowRadius;
    for(int y = 0; y < src->height(); y++)      
      for (int x = 0; x < src->width(); x++){
        M[x][y]=(*src)(x,y);
        M2[x][y]=(*src)(x,y)*(*src)(x,y);
        if(x && y){
          M[x][y]+=M[x-1][y]+M[x][y-1]-M[x-1][y-1];
          M2[x][y]+=M2[x-1][y]+M2[x][y-1]-M2[x-1][y-1];
        }
        else if(x && !y){
          M[x][y]+=M[x-1][y];
          M2[x][y]+=M2[x-1][y];
        }
        else if(!x && y){
          M[x][y]+=M[x][y-1];
          M2[x][y]+=M2[x][y-1];
        }

      }

    for(int y = 0; y < src->height(); y++){
      for (int x = 0; x < src->width(); x++){
        int limInf , limSup , limRight,limLeft = 0;
        int area = 1;

        // We take the limits of the enviroment
        if(x-env < 0){
          limLeft = 0;
        }
        else limLeft=x-env;

        if(x+env >= src->width()){
          limRight = src->width()-1;
        }
        else limRight=x+env;

        if(y-env < 0){
          limSup = 0;
        }
        else limSup=y-env;

        if(y+env >= src->height()){
          limInf = src->height()-1;
        }
        else limInf=y+env;       

        area = (limInf-limSup+1)*(limRight-limLeft+1);

        //Calculate the mean
        double mean = double(M[limLeft][limSup]+ M[limRight][limInf] - M[limLeft][limInf] -M[limRight][limSup])/area;
        double mean2 = double(M2[limLeft][limSup]+ M2[limRight][limInf] - M2[limLeft][limInf] -M2[limRight][limSup])/area;
        //Compute the Standar Deviacion square(Mean^2-mean2)
        double sd = sqrt(mean2-mean*mean);

        //Apply the Threshold T=mean-0.2sd
        float T = mean - k*sd;
        (*result)(x,y) = (*src)(x,y) < T ? 0 : 1;
      }
    }

    for(int i = 0; i < src->width();++i){
      delete []M[i];
      delete []M2[i];

    }
    delete []M;
    delete []M2;

    return result;
  }

  ImageFloat *binarize_sauvola(const ImageFloat *src, int windowRadius, float k, float r)
  {
    april_assert(src->width()  > 0 && "Zero-sized image!");
    april_assert(src->height() > 0 && "Zero-sized image!");
    // Image Integral Matrix
    ImageFloat *result = new ImageFloat(src->width(), src->height());
    double **M = new double*[src->width()];
    for(int i = 0; i < src->width(); ++i)
      M[i] = new double[src->height()];
    // Square Image Integral
    double **M2 = new double*[src->width()];
    for(int i = 0; i < src->width(); ++i)
      M2[i] = new double[src->height()];

    int env=windowRadius;
    for(int y = 0; y < src->height(); y++)      
      for (int x = 0; x < src->width(); x++){
        M[x][y]=(*src)(x,y);
        M2[x][y]=(*src)(x,y)*(*src)(x,y);
        if(x && y){
          M[x][y]+=M[x-1][y]+M[x][y-1]-M[x-1][y-1];
          M2[x][y]+=M2[x-1][y]+M2[x][y-1]-M2[x-1][y-1];
        }
        else if(x && !y){
          M[x][y]+=M[x-1][y];
          M2[x][y]+=M2[x-1][y];
        }
        else if(!x && y){
          M[x][y]+=M[x][y-1];
          M2[x][y]+=M2[x][y-1];
        }

      }

    for(int y = 0; y < src->height(); y++){
      for (int x = 0; x < src->width(); x++){
        int limInf , limSup , limRight,limLeft = 0;
        int area = 1;

        // We take the limits of the enviroment
        if(x-env < 0){
          limLeft = 0;
        }
        else limLeft=x-env;

        if(x+env >= src->width()){
          limRight = src->width()-1;
        }
        else limRight=x+env;

        if(y-env < 0){
          limSup = 0;
        }
        else limSup=y-env;

        if(y+env >= src->height()){
          limInf = src->height()-1;
        }
        else limInf=y+env;       

        area = (limInf-limSup+1)*(limRight-limLeft+1);

        //Calculate the mean
        double mean = double(M[limLeft][limSup]+ M[limRight][limInf] - M[limLeft][limInf] -M[limRight][limSup])/area;
        double mean2 = double(M2[limLeft][limSup]+ M2[limRight][limInf] - M2[limLeft][limInf] -M2[limRight][limSup])/area;
        //Compute the Standar Deviacion square(Mean^2-mean2)
        double sd = sqrt(mean2-mean*mean);

        //Apply the Threshold T=mean-0.2sd
        float T = mean *(1+k*(sd/(r-1)));
        (*result)(x,y) = (*src)(x,y) < T ? 0 : 1;
      }
    }

    for(int i = 0; i < src->width();++i){
      delete []M[i];
      delete []M2[i];

    }
    delete []M;
    delete []M2;

    return result;
  }


  ImageFloat *binarize_otsus(const ImageFloat *src)
  {
    april_assert(src->width()  > 0 && "Zero-sized image!");
    april_assert(src->height() > 0 && "Zero-sized image!");
    //Image Integral Matrix
    ImageFloat *result       = new ImageFloat(src->width(), src->height());

    int ntotal = src->height()*src->width();
    long double h[255];

    int T             = 0;
    long double csum  = 0.0;
    long double sum   = 0.0;
    long double sbmax = -1.0;

    double p1 = 0.0;
    double p2 = 0.0;

    // Prepare the gray histogram
    for (int i = 0; i< 255; i++) h[i] = 0;

    // Calculate the histogram
    for (int y = 1; y < src->height(); y++){
      for (int x = 1; x < src->width(); x++){
        // Samplear to 255 value
        int v = (int)((*src)(x,y)*255);
        h[v]++;
      }
    }

    // Normalize data
    // Calculate sum of g * h[g];
    for (int i = 0; i< 255; i++){
      h[i] = h[i]/ntotal;
      sum  += h[i]*i;
    }

    // For each histogram
    for(int g = 0; g < 255; g++){

      double m1 = 0.0;
      double m2 = 0.0;
      double sb = 0.0;

      p1 += h[g];

      if ( !p1 ) continue;
      p2 = 1 - p1;
      if (!p2) break;

      csum += g*h[g];
      m1 = csum/p1;
      m2 = (sum - csum) / p2;
      sb = p1*p2*(m1-m2)*(m1-m2);

      // Take the best T who maximize inter-class variance
      if(sb > sbmax){
        sbmax = sb;
        T = g;
      }      

    }
    // Apply the new Threshold
    for(int y = 1; y < src->height(); y++)      
      for (int x = 1; x < src->width(); x++){
        int v = (int)((*src)(x,y)*255);
        if( v > T) (*result)(x,y) = 1;
        else (*result)(x,y) = 0;
      } 

    return result;
  }


  ImageFloat *binarize_threshold(const ImageFloat *src, double threshold) {

    april_assert(src->width()  > 0 && "Zero-sized image!");
    april_assert(src->height() > 0 && "Zero-sized image!");
    //Image Integral Matrix
    ImageFloat *result       = new ImageFloat(src->width(), src->height());
  
    // int ntotal = src->height()*src->width();

    for (int y = 0; y < src->height(); y++){
      for (int x = 0; x < src->width(); x++){
        if ((*src)(x,y) < threshold) {
          (*result)(x,y) = 0;   
        }
        else {
          (*result)(x,y) = 1;
        }
      }
    }
    return result;
  }

} // namespace Imaging
