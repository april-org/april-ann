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
#include "geom_param.h"
#include "clamp.h"
#include "derivative.h"
#include <cstdio>
#include <cstring>

using april_utils::clamp;
using april_utils::derivative1;
using april_utils::derivative2;

namespace OCR {
  namespace OffLineTextPreprocessing {
    
    const float GeomParam::THRESHOLD = 0.7;
    
#define SQR(x) ((x)*(x))

    // val = (val - base) * scal;
    void GeomParam::copyNormVector(float *v, MatrixFloat *mat,
				   int ncol,
				   float base, float scal) {
      int n = mat->getDimSize(0);
      int nparams = mat->getDimSize(1);
      april_assert(ncol < nparams);
      for (int i=0; i<n; ++i) {
	(*mat)(i,ncol) = (v[i] - base) * scal;
      }
    }
    
    
    void GeomParam::contours(const ImageFloat *img, int col,
			     float text_center, float *sup, float *inf) {
      // binarizing threshold
      const float THRESHOLD = 0.2;
      int sup_contour = img->height - 1;
      int inf_contour = 0;
      int row = 0;
      float val = (*img)(col, row);
      while ((val < THRESHOLD) && (row < img->height - 1)) {
	++row;
	val = (*img)(col, row);
      }
      sup_contour = row;
      
      row = img->height - 1;
      val = (*img)(col, row);
      while ((val < THRESHOLD) && (row > 0)) {
	--row;
	val = (*img)(col, row);
      }
      inf_contour = row;
      
      // when inferior and superior contours cross, they will be placed in the
      // text center.
      if (sup_contour > inf_contour) {
	sup_contour = inf_contour = int(text_center);
      }
      *sup = float(sup_contour);
      *inf = float(inf_contour);
    }


    // OJO! Este parametrizador asume que 0 = blanco y 1 = black, al contrario que
    // la clase Imagen a julio de 2009 :P
    //
    // params es una cadena que puede contener las siguientes letras:
    // S: contour superior
    // I: contour inferior
    // E: energy
    // P: posicion media de la energy (momento de primer orden)
    // D: desv. tipica de la posicion de la energy (momento de segundo orden)
    //
    // T: numero de trazos
    //
    // Q: derivative1 del c. sup
    // A: derivative1 del c. inf
    //
    // Z: derivative2 del c. sup
    // X: derivative2 del c. inf
    //
    // H: "Altura" -> inf_contour - sup_contour
    // J: Derivative de H
    //
    // M: Derivative de la posicion media de la energy
    // 
    // R: Density de pixels entre el contour superior y el inferior.
    //
    // Si se ponen en minusculas se normalizan respecto a las lineas base
    MatrixFloat* GeomParam::extract(const ImageFloat *img,
				    const char *params) {
      // variance epsilon, considered as zero
      const float EPSILON = 0.01;
      // inferior and upper baselines (from a projection-based algorithm)
      int lower_base, upper_base;
      OCR::OffLineTextPreprocessing::baseLinesRLSA(img,
						   &lower_base, &upper_base);
      
      // WARNING! because coordinates are (x,y) instead of (row,col)
      // the inferior line has a higher y coodinate :P
      float text_height = lower_base - upper_base + 1;
      float text_center = (lower_base + upper_base)/2.0;
	
      int matrix_size[2];
      MatrixFloat *result;

      int NPARAM = strlen(params);
	
      // the matrix has as many rows as columns in the image, and as many
      // columns as parameters
      matrix_size[0] = img->width; 	
      matrix_size[1] = NPARAM;

      result = new MatrixFloat(2, matrix_size);

      // temporal vectors which store all the parameters
      float *v_sup_contour = new float[img->width];
      float *v_inf_contour = new float[img->width];
      float *v_height = new float[img->width];
      float *v_energy = new float[img->width];
      float *v_density = new float[img->width];
      float *v_mean_energy_pos = new float[img->width];
      float *v_stddev_energy = new float[img->width];
      float *v_sup_derivative = new float[img->width];
      float *v_inf_derivative = new float[img->width];
      float *v_derivative2_sup = new float[img->width];
      float *v_derivative2_inf = new float[img->width];
      float *v_derivative_pme = new float[img->width];
      float *v_derivative_height = new float[img->width];
      float *v_orig_nstrokes = new float[img->width];
      float *v_nstrokes = new float[img->width]; // after voting (2 of 3)
	
      // traverse all the columns
      for (int col = 0; col < img->width; ++col) {
	float energy = 0;
	float mean_energy_pos = 0;
	float mean_pos_squared = 0;
	float variance = 0;
	// contours
	contours(img, col, text_center, 
		 &v_sup_contour[col], &v_inf_contour[col]);
	
	int sup_contour = int(v_sup_contour[col]);
	int inf_contour = int(v_inf_contour[col]);
	
	// height
	v_height[col] = inf_contour - sup_contour;
	
	// energy levels, mean position (weighted) and variance
	for (int row = sup_contour; row <= inf_contour; ++row) {
	  float val = (*img)(col, row);
	  energy += val;
	  //float row_norm = (row - upper_base)/text_height;
	  //mean_energy_pos += row_norm * val;
	  //mean_pos_squared += SQR(row_norm) * val;
	  mean_energy_pos += row * val;
	  mean_pos_squared += SQR(row) * val;
	}
	if (energy > 0) {
	  mean_energy_pos /= energy;
	  // var(x) = E(x^2) - (E(x))^2	
	  variance = mean_pos_squared / energy - SQR(mean_energy_pos);
	  if (fabsf(variance) < EPSILON) variance = 0;
	}
	else {
	  mean_energy_pos = text_center;
	  variance = 0;
	}
	v_energy[col] = energy;
	v_density[col] = v_height[col] > 0 ? energy/v_height[col] : 0;
	v_mean_energy_pos[col] = mean_energy_pos;
	v_stddev_energy[col] = sqrtf(variance);
	
	// number of strokes.
	// starts at black and ends at black.
	const float THRESHOLD = 0.8; // energy threshold
	bool black=true;
	int cblack=0;
	int nstrokes=0;
	const int STROKES_THRESHOLD=1;
	for (int row=sup_contour; row<=inf_contour; ++row) {
	  if ( (*img)(col,row) > THRESHOLD ) {
	    // black pixel
	    black=true;
	    cblack++;
	  }
	  else {
	    // white pixel
	    if (black) {
	      if (cblack >= STROKES_THRESHOLD) {
		++nstrokes;
	      }
	      
	      cblack=0;
	    }
	    black = false;
	  }
	}
	if (black && cblack >= STROKES_THRESHOLD)
	  ++nstrokes;
	
	v_orig_nstrokes[col] = nstrokes;
      }
      
      // derivatives
      derivative1(v_sup_contour, v_sup_derivative, img->width);
      derivative1(v_inf_contour, v_inf_derivative, img->width);
      derivative1(v_mean_energy_pos, v_derivative_pme, img->width);
      derivative1(v_height, v_derivative_height, img->width);
      
      derivative2(v_sup_contour, v_derivative2_sup, img->width);
      derivative2(v_inf_contour, v_derivative2_inf, img->width);
      
      // compute strokes by voting
      for (int col=0; col < img->width; ++col) {
	int izq, der;
	
	// nstrokes
	izq = clamp(col - 1, 0, img->width - 1);
	der = clamp(col + 1, 0, img->width - 1);
	if (v_orig_nstrokes[izq] == v_orig_nstrokes[der])
	  v_nstrokes[col] = v_orig_nstrokes[izq];
	else
	  v_nstrokes[col] = v_orig_nstrokes[col];
      }
	
      // put the required parameters in the result matrix
      int i=0;
      while (params[i]) {
	float inv_text_height = 1.0/text_height;
	switch(params[i]) {
	case 'S':
	  result->putCol(i, v_sup_contour, img->width);
	  break;
	case 's':
	  copyNormVector(v_sup_contour, result, i, upper_base, inv_text_height);
	  break;
	case 'I':
	  result->putCol(i, v_inf_contour, img->width);
	  break;
	case 'i':
	  copyNormVector(v_inf_contour, result, i, upper_base, inv_text_height);
	  break;
	case 'E':
	  result->putCol(i, v_energy, img->width);
	  break;
	case 'e':
	  copyNormVector(v_energy, result, i, 0, inv_text_height);
	  break;
	case 'P':
	  result->putCol(i, v_mean_energy_pos, img->width);
	  break;
	case 'p':
	  copyNormVector(v_mean_energy_pos, result, i, upper_base, inv_text_height);
	  break;
	case 'D':
	  result->putCol(i, v_stddev_energy, img->width);
	  break;
	case 'd':
	  copyNormVector(v_stddev_energy, result, i, 0, inv_text_height);
	  break;
	case 'Q':
	  result->putCol(i, v_sup_derivative, img->width);
	  break;
	case 'q':
	  copyNormVector(v_sup_derivative, result, i, 0, inv_text_height);
	  break;
	case 'A':
	  result->putCol(i, v_inf_derivative, img->width);
	  break;
	case 'a':
	  copyNormVector(v_inf_derivative, result, i, 0, inv_text_height);
	  break;
	case 'T':
	case 't': //TODO: Normalizar?
	  copyNormVector(v_nstrokes, result, i, 0, 0.3);
	  break;
	case 'H':
	  result->putCol(i, v_height, img->width);
	  break;
	case 'h':
	  copyNormVector(v_height, result, i, 0, inv_text_height);
	  break;
	case 'J':
	  result->putCol(i, v_derivative_height, img->width);
	  break;
	case 'j':
	  copyNormVector(v_derivative_height, result, i, 0, inv_text_height);
	  break;
	case 'M':
	  result->putCol(i, v_derivative_pme, img->width);
	  break;
	case 'm':
	  copyNormVector(v_derivative_pme, result, i, 0, inv_text_height);
	  break;
	case 'Z':
	  result->putCol(i, v_derivative2_sup, img->width);
	  break;
	case 'z':
	  copyNormVector(v_derivative2_sup, result, i, 0, inv_text_height);
	  break;
	case 'X':
	  result->putCol(i, v_derivative2_inf, img->width);
	  break;
	case 'x':
	  copyNormVector(v_derivative2_inf, result, i, 0, inv_text_height);
	  break;
	case 'R':
	case 'r':
	  result->putCol(i, v_density, img->width);
	  break;
	default:
	  ERROR_EXIT1(128,
		      "param: invalid character '%c' in parameters string!\n",
		      params[i]);
	}
	++i;
      }

      delete [] v_sup_contour;
      delete [] v_inf_contour;
      delete [] v_height;
      delete [] v_energy;
      delete [] v_mean_energy_pos;
      delete [] v_stddev_energy;
      delete [] v_sup_derivative;
      delete [] v_inf_derivative;
      delete [] v_derivative2_sup;
      delete [] v_derivative2_inf;
      delete [] v_derivative_pme;
      delete [] v_derivative_height;
      delete [] v_orig_nstrokes;
      delete [] v_nstrokes;
	
	
      return result;
    }


  }
}
