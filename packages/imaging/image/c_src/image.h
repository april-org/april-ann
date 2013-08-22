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
#ifndef IMAGE_H
#define IMAGE_H

#include "referenced.h"
#include "matrix.h"
#include "affine_transform.h"
#include <cmath>

template <typename T>
class Image : public Referenced {
 public:
  Matrix<T> *matrix; // dimension 2 is assumed
  int offset;
  int width,height;
  //Constructors
  Image(Matrix<T> *mat);
  Image(Matrix<T> *mat, 
	int width, int height,
	int offset_w, int offset_h);
  Image(int width, int height, T value=T()); // Image with a new Matrix, filled with value
  Image(Image &other); // copy constructor
  //Destructor...
  virtual ~Image();
  //Methods
  T& operator () (int x, int y) { 
    return matrix->getRawDataAccess()->getPPALForReadAndWrite()[offset+x+y*matrix_width()];
  }
  
  T operator () (int x, int y) const {
    return matrix->getRawDataAccess()->getPPALForRead()[offset+x+y*matrix_width()];
  }

  // Bound-checking version of operator()
  T getpixel(int x, int y, T default_value) const {
    if (x>=0 && y>=0 && x<width && y<height) return (*this)(x,y);
    else return default_value;
  }

  T getpixel_bilinear(float x, float y, T default_value) const {
    float fx = fabsf(x - trunc(x));
    float fy = fabsf(y - trunc(y));
    float dx = (x >= 0.0f ? 1.0f : -1.0f);
    float dy = (y >= 0.0f ? 1.0f : -1.0f);
    T h1 = (1-fx)*getpixel(int(x), int(y), default_value) + fx*getpixel(int(x+dx), int(y), default_value);
    T h2 = (1-fx)*getpixel(int(x), int(y+dy), default_value) + fx*getpixel(int(x+dx), int(y+dy), default_value);
    return (1-fy)*h1 + fy*h2;
  }
  
  int matrix_width()  const { return matrix->getDimSize(1); }
  int matrix_height() const { return matrix->getDimSize(0); }
  int offset_width()  const { return offset % matrix_width(); }
  int offset_height() const { return offset / matrix_width(); }
  Image<T> *clone() const;
  Image<T> *crop(int width, int height,
	      int offset_w, int offset_h) const;  
  Image<T> *clone_subimage(int width, int height,
			   int offset_w, int offset_h,
			   T default_color) const;  
  Image<T> *crop_with_padding(int width, int height,
			      int offset_w, int offset_h,
			      T default_color) const;  
  void projection_v(T *v) const ; // v is not created here
  void projection_v(Matrix<T> **m) const; // creates matrix m
  void projection_h(T *v) const ; // v is not created here
  void projection_h(Matrix<T> **m) const; // creates matrix m
  Image<T> *shear_h(double radians, T default_color) const;
  void shear_h_inplace(double radians, T default_color);
  
  void min_bounding_box(float threshold, int *w, int *h, int *x, int *y) const;
  void copy(const Image<T> *src, int dst_x, int dst_y);
  
  Image<T> *rotate90_cw() const; // Rotate 90 degrees clockwise, creates a new image
  Image<T> *rotate90_ccw() const; // Rotate 90 degrees counter-clockwise, new image
  Image<T> *resize(int dst_width, int dst_height) const; // Resize an image
  Image<T> *invert_colors() const; // Invert the color of every pixel, new image

  Image<T> *convolution5x5(float *k, T default_color=T()) const;
  Image<T> *affine_transform(AffineTransform2D *trans, T default_value, int *offset_x=0, int *offset_y=0) const;
  
  Image<T> *remove_blank_columns() const;
  Image<T> *add_rows(int top_rows, int bottom_rows, T value) const;
  Image<T> *substract_image(Image<T> *img, T low, T high) const;

  void threshold_image(T low, T high, T value_low, T value_high);
 private:
  void invert_affine_matrix(float c[6], float dest[6]) const;
};

/*** Implementacion ***/
#include "image.cc"

#endif // IMAGE_H
