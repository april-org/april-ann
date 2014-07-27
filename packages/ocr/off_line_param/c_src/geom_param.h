/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Salvador España-Boquera, Jorge Gorbe-Moya, Francisco
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
#ifndef GEOM_PARAM_H
#define GEOM_PARAM_H

#include "matrixFloat.h"
#include "utilImageFloat.h"

namespace OCR {
  namespace OffLineTextPreprocessing {
    
    template <typename T> 
    void RLSA(Image<T> *img, int threshold) {
      const float BINARIZING_THRESHOLD = 0.5f;
      // for each image row, seek white pixel runs between two blacks. If the
      // run length is inferior to a threshold, the run will be filled with
      // black.
      for (int row=0; row < img->height(); ++row) {
	int left=-1;
	int right=0;
	int run_length=0;
	while (right < img->width()) {
	  // a black pixel
	  if ((*img)(right, row) > BINARIZING_THRESHOLD) {
	    if (left != -1) {
	      if ((run_length > 0) && (run_length < threshold)) {
		for(int col=left; col<=right; ++col)
		  (*img)(col,row) = 1.0; // FIXME:CTENEGRO
	      }
	    }
	    run_length=0; 
	    left=right;
	  }
	  else {
	    // a white pixel
	    ++run_length;
	  }
	  ++right;
	}
      }
    }
    
    template<typename T> void baseLinesRLSA(const Image<T> *img,
					    int *lower, int *upper,
					    const float projection_th=0.5f) {
      // if the projection is inferior to this threshold, it won't be placed
      // between text baselines
      const int RLSA_TH=30;
      
      // work copy
      Image<T> *i2=img->clone();
      
      RLSA(i2, RLSA_TH);
      T *projection = new T[img->height()];
      i2->projection_h(projection);
      
      T maximum = 0;
      for (int i=0; i < img->height(); ++i) {
	if (projection[i] > maximum)
	  maximum = projection[i];
      }
      int u=0;
      while (u < img->height()) {
	if (projection[u] > projection_th * maximum)
	  break;
	++u;
      }
      int l = img->height() - 1;
      while (l >= 0) {
	if (projection[l] > projection_th * maximum)
	  break;
	--l;
      }
      *upper = u;
      *lower = l;
      delete i2;
      delete[] projection;
    }
    
    class GeomParam {
    public:
      /// Threshold which defines when the pixel energy is considered ink
      static const float THRESHOLD;

      static MatrixFloat *extract (const ImageFloat *i, const char *params);
      
    private:
      static void contours(const ImageFloat *img, int col,
			   float text_center, float *sup, float *inf);
      
      
      static void copyNormVector(float *v, MatrixFloat *mat, int ncol,
				 float base, float scal);
    };
    
  }
}

#endif // GEOM_PARAM_H
