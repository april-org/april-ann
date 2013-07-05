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
/*
 *    Fichero de implementacion (incluido por la cabecera image.h)
 *
 */

#ifndef _IMAGE_CC_
#define _IMAGE_CC_

#include "image.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "clamp.h"
#include "maxmin.h"

template <typename T>
Image<T>::Image(Matrix<T> *mat) {
  if (!mat->isSimple())
    ERROR_EXIT(128, "Image only works with simple matrices "
	       "(contiguous, and in row-major\n");
  // if (mat->numDim != 2) { ... } // <- TODO
  matrix = mat;
  IncRef(matrix); // garbage collection
  offset = 0;
  width = matrix_width();
  height = matrix_height();
}

template <typename T>
Image<T>::Image(Matrix<T> *mat, 
		int width, int height,
		int offset_w, int offset_h) {
  if (!mat->isSimple())
    ERROR_EXIT(128, "Image only works with simple matrices "
	       "(contiguous, and in row-major\n");
  matrix = mat;
  IncRef(matrix); // garbage collection
  offset = offset_w + offset_h*matrix_width();
  this->width  = width;
  this->height = height;
}

template <typename T>
Image<T>::Image(Image &other) { // copy constructor
  matrix = other.matrix;
  offset = other.offset;
  width  = other.width;
  height = other.height;
  IncRef(matrix);
}

template <typename T>
Image<T>::Image(int width, int height, T value)
  :width(width), height(height)
{
  offset = 0;
  
  int dims[2];
  dims[0] = height;
  dims[1] = width;
  matrix = new Matrix<T>(2,dims, value);
  IncRef(matrix);
}


template <typename T>
Image<T>::~Image() {
  DecRef(matrix); // garbage collection
}

template <typename T>
Image<T>* Image<T>::crop(int width, int height,
			 int offset_w, int offset_h) const {
  assert("Cropping region must be contained in source image" &&
	 (offset_w >= 0 && offset_h >=0 &&
	  offset_w+width <= this->width && 
	  offset_h+height <= this->height));

  return new Image<T>(matrix,
		      width, height,
		      offset_w + offset_width(),
		      offset_h + offset_height());
}

template <typename T>
Image<T>* Image<T>::clone_subimage(int w, int h,
				   int offset_w, int offset_h,
				   T default_color) const {
  int dims[2];
  dims[0] = h;
  dims[1] = w;
  Matrix<T> *mat = new Matrix<T>(2,dims,default_color);
  Image<T>  *img = new Image<T>(mat);

  // averiguar tamanyo subimage a copiar:
  int x_dest=0,        y_dest=0;
  int w_crop=w,        h_crop=h;
  int x_crop=offset_w, y_crop=offset_h;

  if (offset_w < 0) { // izquierda
    x_crop = 0;
    w_crop -= -offset_w;
    x_dest = offset_w;
  }
  if (offset_w + w > width) { // derecha
    w_crop -= offset_w + w - width;
  }
  if (offset_h < 0) { // arriba
    y_crop = 0;
    h_crop -= -offset_h;
    y_dest = offset_h;
  }
  if (offset_h + h > height) { // abajo
    h_crop -= offset_h + h - height;
  }
  if (w_crop > 0 && h_crop > 0) {
    img->copy(crop(w_crop,h_crop,x_crop,y_crop), x_dest, y_dest);
  }
  return img;
}

template <typename T>
Image<T>* Image<T>::crop_with_padding(int width, int height,
				      int offset_w, int offset_h,
				      T default_color) const {

  if (offset_w >= 0 && offset_h >=0 &&
      offset_w+width  <= this->width && 
      offset_h+height <= this->height)
    return crop(width, height, offset_w, offset_h);
  else
    return clone_subimage(width, height, offset_w, offset_h, default_color);
}


template <typename T>
Image<T>* Image<T>::clone() const {
  int dims[2];
  dims[0] = height;
  dims[1] = width;
  Matrix<T> *mat = new Matrix<T>(2,dims);
  Image<T>  *img = new Image<T>(mat);
  //
  // --more elegant but less efficient--
  //   for (int y=0;y<height;y++)
  //     for (int x=0;x<width;x++)
  //       img(x,y) = operator()(x,y);
  // 
  T *dest   = mat->getRawDataAccess()->getPPALForReadAndWrite();
  const T *source = matrix->getRawDataAccess()->getPPALForRead() + offset;
  for (int y=0;y<height;y++) {
    for (int x=0;x<width;x++)
      dest[x] = source[x];
    source += matrix_width();
    dest   += width;
  }
  return img;
}

template <typename T>
void Image<T>::projection_v(T *v) const {
  // v has size width
  for (int x=0;x<width;x++)
    v[x] = 0;
  const T *source = matrix->getRawDataAccess()->getPPALForRead() + offset;
  for (int y=0;y<height;y++) {
    for (int x=0;x<width;x++)
      v[x] += source[x];
    source += matrix_width();
  }    
}

template <typename T>
void Image<T>::projection_v(Matrix<T> **m) const {
  int dims[1];
  dims[0] = width;
  *m = new Matrix<T>(1,dims);
  projection_v((*m)->getRawDataAccess()->getPPALForWrite());
}

template <typename T>
void Image<T>::projection_h(T *v) const {
  // v has size height
  const T *source = matrix->getRawDataAccess()->getPPALForRead() + offset;
  for (int y=0;y<height;y++) {
    T aux = 0;
    for (int x=0;x<width;x++)
      aux += source[x];
    v[y] = aux;
    source += matrix_width();
  }
}

template <typename T>
void Image<T>::projection_h(Matrix<T> **m) const {
  int dims[1];
  dims[0] = height;
  *m = new Matrix<T>(1,dims);
  projection_h((*m)->getRawDataAccess()->getPPALForWrite());
}

/**

   -  angle va en radianes
   -  default_value es el color de los pixels nuevos que aparecen
   en las esquinas
  
   +-------+  +-------+
   |       |  |\       \
   |  OLD  |  | \  NEW  \
   |       |  |an\       \
   +-------+  |gle+-------+
*/
template <typename T>
Image<T>* Image<T>::shear_h(double angle, T default_value) const {
  int dims[2];
  //printf("angle = %frad = %fdeg\n",angle, angle*180/M_PI);
  dims[0]=height;
  if (angle > 0)
    dims[1]=width+int(height*tan(angle))+1;
  else
    dims[1]=width+int(height*tan(-angle))+1;

  //printf("dims = %dfilas x %dcolumnas\n", dims[0], dims[1]);
	
  Matrix<T> *mat = new Matrix<T>(2,dims);
  Image<T>  *img = new Image<T>(mat);

	
  if (angle > 0)
    {
      // Angle > 0
      const T *source_line = matrix->getRawDataAccess()->getPPALForRead() + offset;
      T *dest_line = mat->getRawDataAccess()->getPPALForReadAndWrite();
      for (int line=0; line<height; line++){
	float x = line*tan(angle);
	float izq = x-int(x);
	float der = 1.0-izq;
	//printf("x=%f izq=%f der=%f\n", x, izq, der);
	int x_int = int(x);	
	// x contiene la posicion del pixel i-esimo
	// por tanto, debemos poner en blanco los pixels [0,x[
	// copiar la fila original en [x, x+width] y seguir
	// con blanco hasta el final

	for (int i=0; i < x_int; i++)
	  dest_line[i]=default_value;
			
	// El primer pixel lo tratamos de forma "especial"
	// porque se obtiene a partir del primer pixel origen y el
	// color por defecto

	dest_line[x_int]=izq*default_value+der*source_line[0];

	for (int i=1; i < width; i++)
	  {
	    dest_line[x_int+i]=izq*source_line[i-1]+der*source_line[i];
	  }

	dest_line[x_int+width]=izq*source_line[width-1]+der*default_value;

			
	for (int i=x_int+width+1; i<dims[1]; i++)
	  dest_line[i]=default_value;
			
	source_line += matrix_width();
	dest_line += dims[1];
      }
    } else {
    // Angle < 0
    // Empezamos por abajo
    angle=-angle;
    const T *source_line = matrix->getRawDataAccess()->getPPALForRead() + offset + matrix_width()*(height-1);
    T *dest_line = mat->getRawDataAccess()->getPPALForReadAndWrite() + dims[1]*(height-1);
    for (int line= 0 ; line < height; line++){
      float x = line*tan(angle);
      float izq = x-int(x);
      float der = 1.0-izq;
      int x_int = int(x);	
      // x contiene la posicion del pixel i-esimo
      // por tanto, debemos poner en blanco los pixels [0,x[
      // copiar la fila original en [x, x+width] y seguir
      // con blanco hasta el final

      for (int i=0; i < x_int; i++)
	dest_line[i]=default_value;
			
      // El primer pixel lo tratamos de forma "especial"
      // porque se obtiene a partir del primer pixel origen y el
      // color por defecto

      dest_line[x_int]=izq*default_value+der*source_line[0];

      for (int i=1; i < width; i++)
	dest_line[x_int+i]=izq*source_line[i-1]+der*source_line[i];

      dest_line[x_int+width]=izq*source_line[width-1]+der*default_value;
			
      for (int i=x_int+width+1; i<dims[1]; i++)
	dest_line[i]=default_value;
			
      source_line -= matrix_width();
      dest_line -= dims[1];
    }
  }

  return img;
}

template <typename T>
void Image<T>::shear_h_inplace(double angle, T default_value) {

  // inc_width contiene el incremento de tam. en x al hacer el shear
  // segun el signo de angle deberemos desplazar el resultado hacia
  // un lado u otro
  int inc_width; 
	
  inc_width = int(height * tan(angle) + 1);
	
	

  if (angle > 0)
    {
      // Angle > 0
      // Tenemos que desplazar la imagen resultante hacia la izq.
      int old_width = width;
      offset -= inc_width;	
      width  += inc_width;
		
      // Ahora la linea de origen y destino es la misma
      // Empezamos por el final
      T *source_line = matrix->getRawDataAccess()->getPPALForReadAndWrite() + offset + (height-1) * matrix_width();
      T *dest_line = source_line;
		
      for (int line=0; line<height; line++){
	float x = line*tan(angle);
	float der = x-int(x);
	float izq = 1.0-der;
	//printf("x=%f izq=%f der=%f\n", x, izq, der);
	int x_int = int(x)+1;	
	// x contiene la posicion del pixel i-esimo
	// por tanto, debemos copiar la fila original
	// desde [inc_width, width[
	// hasta [inc_width - x, width-x[
	// y luego poner en blanco los pixels
	// en el intervalo [width-x, width] 

			
	// El primer pixel lo tratamos de forma "especial"
	// porque se obtiene a partir del primer pixel origen y el
	// color por defecto

	dest_line[inc_width-x_int]=izq*default_value+der*source_line[inc_width];

	for (int i=1; i < old_width; i++)
	  dest_line[(inc_width-x_int)+i]=izq*source_line[inc_width+i-1]+der*source_line[inc_width+i];

	dest_line[(inc_width-x_int)+old_width]=izq*source_line[inc_width+old_width-1]+der*default_value;
			
	for (int i=width-x_int+1; i<width; i++)
	  dest_line[i]=default_value;
			
	source_line -= matrix_width();
	dest_line -= matrix_width();
      }
    } else {
    // Angle < 0
    //
    // Ahora tenemos que ir copiando las filas de der a izq
    // para no machacar la parte de la fila que aun no hemos usado
    //
    // La imagen queda ahora inclinada hacia la derecha
    int old_width=width;
    angle = -angle;
    width += inc_width;
		
    T *source_line = matrix->getRawDataAccess()->getPPALForReadAndWrite() + offset + (height - 1) * matrix_width();
    T *dest_line = source_line;
		
    for (int line = 0 ; line < height; line++){
      float x = line*tan(angle);
      float izq = x-int(x);
      float der = 1.0-izq;
      int x_int = int(x);	
      // x contiene la posicion del pixel i-esimo
      // yendo de derecha a izq debemos copiar
      // desde [0,old_width] hasta [x, old_width+x]
      // y luego rellenar [0, x[ con el color por defecto

			
      // Copiamos el ultimo pixel
      dest_line[old_width+x_int]=izq*source_line[old_width-1]+der*default_value;

      // Los de en medio
      for (int i=old_width-1; i >= 1; i--)
	dest_line[x_int+i]=izq*source_line[i-1]+der*source_line[i];

      // El primero
      dest_line[x_int]=izq*default_value+der*source_line[0];
			
      // y rellenamos el resto con el color por defecto
      for (int i=x_int-1; i>=0; i--)
	dest_line[i]=default_value;
			
      source_line -= matrix_width();
      dest_line -= matrix_width();
    }
		
  }
		
  return;
}

// FIXME: Mover a ocr
template <typename T>
void Image<T>::min_bounding_box(float threshold, int *w, int *h, int *x, int *y) const
{
  // Buscamos el primer pixel no-blanco (inferior al umbral) de la imagen
  // desde arriba, abajo, izquierda y derecha y hacemos un crop.
  int upper = 0, lower = height - 1, left = 0, right = width - 1;
  int row, col;
  bool found;
  const float *d;

  // Desde arriba
  found = false;
  row = 0;
  d = matrix->getRawDataAccess()->getPPALForRead();
  while (row < height && !found)
    {
      for (col = 0; col < width; ++col)
	{
	  T val = d[offset + row*matrix_width() + col];
	  if (val < threshold)
	    {
	      upper = row;
	      found = true;
	    }
	}

      ++row;
    }

  // Desde abajo
  row = height - 1;
  found = false;
  while (row >= 0 && !found)
    {
      for (col = 0; col < width; ++col)
	{
	  T val = d[offset + row*matrix_width() + col];                  
	  if (val < threshold)
	    {
	      lower = row;
	      found = true;
	    }

	}

      --row;
    }

  // Desde la izquierda
  col = 0;
  found = false;
  while (col < width && !found)
    {
      for (row = 0; row < height; ++row)
	{
	  T val = d[offset + row*matrix_width() + col]; 
	  if (val < threshold)
	    {
	      left = col;
	      found = true;
	    }
	}

      ++col;
    }

  // Desde la derecha
  col = width - 1;
  found = false;
  while (col >= 0 && !found)
    {
      for (row = 0; row < height; ++row)
	{
	  T val = d[offset + row*matrix_width() + col]; 
	  if (val < threshold)
	    {
	      right = col;
	      found = true;
	    }
	}
      --col;
    }

  //printf("upper=%d, lower=%d, left=%d, right=%d\n", upper, lower, left, right);

  *w = right - left + 1;
  *h = lower - upper + 1;
  *x = left;
  *y = upper;
}

// Copia la imagen src a partir de las coordenadas (dst_x, dst_y)
// Recorta si es necesario
template<typename T>
void Image<T>::copy(const Image<T> *src, int dst_x, int dst_y)
{
  int x0=0, y0=0;

  if (dst_x < 0) x0 = -dst_x;
  if (dst_y < 0) y0 = -dst_y;

  for (int y=y0; (y < src->height) && (y+dst_y < height); ++y)
    for (int x=x0; (x < src->width) && (x+dst_x < width); ++x) 
      (*this)(x+dst_x, y+dst_y)=(*src)(x, y);
}


template<typename T>
Image<T> *Image<T>::rotate90_cw() const
{
  int dimensions[2];
  dimensions[0] = width; 
  dimensions[1] = height; 

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  for (int y=0; y < height; ++y)
    {
      for (int x=0; x < width; ++x)
	{
	  //printf("(%d, %d) ---> (%d, %d)\n",x,y,height-1-y,x);
	  (*result)(height - 1 - y, x)=(*this)(x, y);
	}
    }
	
  return result;
}

template<typename T>
Image<T> *Image<T>::rotate90_ccw() const
{
  int dimensions[2];
  dimensions[0] = width; 
  dimensions[1] = height; 

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  for (int y=0; y < height; ++y)
    {
      for (int x=0; x < width; ++x)
	{
	  (*result)(y, width - 1 - x)=(*this)(x, y);
	}
    }
	
  return result;
}


// Warning: This operation is not clearly defined for images
// outside the range [0.0, 1.0]
template<typename T>
Image<T> *Image<T>::invert_colors() const
{
  const T WHITE(1.0f);

  int dimensions[2];
  dimensions[0] = height; 
  dimensions[1] = width; 

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  for (int y=0; y < height; ++y)
    {
      for (int x=0; x < width; ++x)
	{
	  (*result)(x, y) = WHITE - (*this)(x, y);
	}
    }
	
  return result;
}


// FIXME: Mover a ocr
template<typename T>
Image<T> *Image<T>::remove_blank_columns() const
{
  const float UMBRAL_BINARIZADO = 0.5f;
  // Contamos las columnas en blanco de la imagen original
  int nblanco=0;
  for (int x=0; x<width; ++x)
    {
      bool blanco=true;
      for(int y=0; y<height; ++y)
	{
	  if ((*this)(x,y)<UMBRAL_BINARIZADO)
	    {
	      blanco=false;
	      break;
	    }
	}

      if (blanco) ++nblanco;
    }

  int dimensions[2];
  dimensions[0]=height;
  dimensions[1]=width - nblanco;
	
  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);
	
  // Ahora copiamos columna a columna
  int xdest=0;
  for (int x=0; x<width; ++x)
    {
      bool blanco=true;
      for (int y=0; y<height;++y)
	{
	  if ((*this)(x,y) < UMBRAL_BINARIZADO)			
	    {
	      blanco=false;
	    }

	  (*result)(xdest, y) = (*this)(x,y);
	}

      if (!blanco) ++xdest;
      if (xdest == width-nblanco) break; // we're finished
    }

  return result;
}

/// Add top_rows at top of the images and bottom_rows at the bottom
template<typename T>
Image<T> *Image<T>::add_rows(int top_rows, int bottom_rows, T value) const
{
  // Contamos las columnas en blanco de la imagen original
  int dimensions[2];
  int new_height = height+top_rows+bottom_rows; 
  dimensions[0] = new_height;
  dimensions[1] = width;

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  // Copy row by row
  int top_image = top_rows;
  int bottom_image = top_rows + height;
    
  for(int y = 0; y < top_image; ++y) {
    for (int x = 0; x < width; x++) {
      (*result)(x,y) = value;
  
    }
  } 
  for(int y = bottom_image; y < new_height; ++y) {
    for (int x = 0; x < width; x++) {
      (*result)(x,y) = value;
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      (*result)(x, top_image+y) = (*this)(x,y);
    }
  }

  return result;
}

/**
 * Aplica un kernel de convolucion de 5x5 a la imagen. Devuelve el resultado
 * en una imagen nueva. El kernel se organiza por filas:
 *
 *                  0  1  2  3  4
 *                  5  6  7  8  9
 *                 10 11 12 13 14
 *                 15 16 17 18 19
 *                 20 21 22 23 24
 */
template<typename T>
Image<T> *Image<T>::convolution5x5(float *k, T default_color) const
{
  int dimensions[2];
  dimensions[0] = height; 
  dimensions[1] = width; 

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      T value = getpixel(x-2, y-2, default_color) * k[0] + 
	getpixel(x-1, y-2, default_color) * k[1] +
	getpixel(x  , y-2, default_color) * k[2] +
	getpixel(x+1, y-2, default_color) * k[3] +
	getpixel(x+2, y-2, default_color) * k[4] +
	getpixel(x-2, y-1, default_color) * k[5] + 
	getpixel(x-1, y-1, default_color) * k[6] +
	getpixel(x  , y-1, default_color) * k[7] +
	getpixel(x+1, y-1, default_color) * k[8] +
	getpixel(x+2, y-1, default_color) * k[9] +
	getpixel(x-2, y,   default_color) * k[10] + 
	getpixel(x-1, y,   default_color) * k[11] +
	getpixel(x  , y,   default_color) * k[12] +
	getpixel(x+1, y,   default_color) * k[13] +
	getpixel(x+2, y,   default_color) * k[14] +
	getpixel(x-2, y+1, default_color) * k[15] + 
	getpixel(x-1, y+1, default_color) * k[16] +
	getpixel(x  , y+1, default_color) * k[17] +
	getpixel(x+1, y+1, default_color) * k[18] +
	getpixel(x+2, y+1, default_color) * k[19] +
	getpixel(x-2, y+2, default_color) * k[20] + 
	getpixel(x-1, y+2, default_color) * k[21] +
	getpixel(x  , y+2, default_color) * k[22] +
	getpixel(x+1, y+2, default_color) * k[23] +
	getpixel(x+2, y+2, default_color) * k[24];
      (*result)(x,y) = value;
    }
  }

  return result;
}

template<typename T>
Image<T> *Image<T>::resize(int dst_width, int dst_height) const
{
  T default_value=T();
  int dimensions[2];
  dimensions[0] = dst_height;
  dimensions[1] = dst_width;

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  // we go through the destination image
  for (int y=0; y<dst_height; y++) {
    for (int x=0; x<dst_width; x++) {
      // each pixel (x,y) in the destination image corresponds to a rectangle (x0,y0)-(x1,y1) in the source image
      float x0 = (float(x)/float(dst_width)) * (width-1);
      float x1 = (float(x+1)/float(dst_width)) * (width-1);
      float y0 = (float(y)/float(dst_height)) * (height-1);
      float y1 = (float(y+1)/float(dst_height)) * (height-1);
      int ix0 = int(x0);
      int ix1 = int(x1);
      int iy0 = int(y0);
      int iy1 = int(y1);

      float area = ((x1-x0)*(y1-y0));
      T sum=T();

      if (iy0 == iy1) {
	// Less than one row (only a fraction)
	float yinterp = 0.5f*(y0+y1);
	if (ix0 == ix1) {
	  sum = (x1-x0) * getpixel_bilinear(0.5f*(x0+x1), yinterp, default_value);
	} 
	else {
	  sum += (float(ix0+1)-x0) * getpixel_bilinear(0.5f*(float(ix0+1)+x0), yinterp, default_value); // beginning of iy0 (possibly fractional)
	  sum += (x1-floor(x1)) * getpixel_bilinear(0.5f*(x1+floor(x1)), yinterp, default_value);    // end of iy0 (possibly fractional)
	  for (int col=ix0+1; col < ix1; col++) {
	    sum += getpixel_bilinear(col+0.5f, yinterp, default_value);
	  }
	}
	sum *= (y1-y0);
      } 
      else {
	// Several rows
	for (int row = iy0; row <= iy1; row++) {
	  float row_fraction, yinterp;
	  T sum_row=T();
	  if (row == iy0) {
	    row_fraction = float(iy0+1)-y0;
	    yinterp = 0.5f*(float(iy0+1)+y0);
	  }
	  else if (row == iy1) {
	    row_fraction = y1-floor(y1);
	    yinterp = 0.5f*(float(y1+floor(y1)));
	  }
	  else {
	    row_fraction=1.0f;
	    yinterp = row+0.5f;
	  }

	  //printf("row_fraction=%f\n", row_fraction);

	  if (ix0 == ix1) {
	    sum_row = (x1-x0) * getpixel_bilinear(0.5f*(x0+x1), row, default_value);
	  } 
	  else {
	    sum_row += (float(ix0+1)-x0) * getpixel_bilinear(0.5f*(float(ix0+1)+x0), yinterp, default_value); // beginning of iy0 (possibly fractional)
	    sum_row += (x1-floor(x1)) * getpixel_bilinear(0.5f*(x1+floor(x1)), yinterp, default_value);    // end of iy0 (possibly fractional)

	    for (int col=ix0+1; col < ix1; col++) {
	      sum_row += getpixel_bilinear(col+0.5f, yinterp, default_value);
	    }
	  }
	  sum += row_fraction*sum_row;
	  //printf("sum_row = %f\n", sum_row);
	}
      }

      (*result)(x,y) = sum/area;
      //printf("(%d, %d) ---> (%f, %f)-(%f, %f) area= %f, sum=%f\n", x, y, x0, y0, x1, y1, area, sum); 
    }
  }
  return result;
}

// c and dest must have 6 elements
template <typename T>
void Image<T>::invert_affine_matrix(float *c, float *dest) const
{
  float det = c[0]*c[4] - c[1]*c[3];
  assert("Affine matrix has no inverse" && det!=0);
  dest[0] =  c[4]/det;
  dest[1] = -c[1]/det;
  dest[3] = -c[3]/det;
  dest[4] =  c[0]/det;
  dest[2] = -dest[0]*c[2] - dest[1]*c[5];
  dest[5] = -dest[3]*c[2] - dest[4]*c[5];
}


/** 
 *  Applies an affine transform given by the matrix
 *
 *  | c[0] c[1] c[2] |
 *  | c[3] c[4] c[5] |
 *  |   0    0    1  |
 *
 *
 */

template <typename T>
Image<T> *Image<T>::affine_transform(AffineTransform2D *trans, T default_value, 
				     int *offset_x, int *offset_y) const
{
  using april_utils::max;
  using april_utils::min;

  float *inverse = trans->getRawDataAccess()->getPPALForReadAndWrite();
  
  float c[6];
  invert_affine_matrix(inverse, c);

  /*
    printf("--transform-------\n");
    printf("%1.3f %1.3f %1.3f\n", c[0], c[1], c[2]);
    printf("%1.3f %1.3f %1.3f\n", c[3], c[4], c[5]);
    printf("0     0     1    \n");

    printf("--inverse transform-------\n");
    printf("%1.3f %1.3f %1.3f\n", inverse[0], inverse[1], inverse[2]);
    printf("%1.3f %1.3f %1.3f\n", inverse[3], inverse[4], inverse[5]);
    printf("0     0     1    \n");
  */
  // New image corners -> apply the inverse to points 0, 1, 2, 3
  //
  //  0 +-----+ 1
  //    |     |
  //  2 +-----+ 3
  //
  int x0 = int(roundf(inverse[2]));
  int y0 = int(roundf(inverse[5]));
  int x1 = int(roundf((width-1) *inverse[0] + inverse[2]));
  int y1 = int(roundf((width-1) *inverse[3] + inverse[5]));
  int x2 = int(roundf((height-1)*inverse[1] + inverse[2]));
  int y2 = int(roundf((height-1)*inverse[4] + inverse[5]));
  int x3 = int(roundf((width-1) *inverse[0] + (height-1)*inverse[1] + inverse[2]));
  int y3 = int(roundf((width-1) *inverse[3] + (height-1)*inverse[4] + inverse[5]));

  int xmax = int(roundf(max(x0, max(x1, max(x2, x3)))));
  int xmin = int(roundf(min(x0, min(x1, min(x2, x3)))));
  int ymax = int(roundf(max(y0, max(y1, max(y2, y3)))));
  int ymin = int(roundf(min(y0, min(y1, min(y2, y3)))));

  int dst_width = xmax-xmin+1;
  int dst_height = ymax-ymin+1;

 
  /*
    fprintf(stderr,"p0=(%d, %d) p1=(%d, %d) p2=(%d, %d) p3=(%d, %d) dst_width = %d, dst_height=%d x: %d-%d, y: %d-%d\n",
    x0, y0, x1, y1, x2, y2, x3, y3, dst_width, dst_height, xmin, xmax, ymin, ymax);
  */

  int dimensions[2];
  dimensions[0] = dst_height;
  dimensions[1] = dst_width;

  Matrix<T> *new_mat = new Matrix<T>(2, dimensions);
  Image<T> *result = new Image<T>(new_mat);

  for (int y=ymin; y<=ymax; y++) {
    for (int x=xmin; x<=xmax; x++) {
      float srcx = c[0]*x+c[1]*y+c[2];
      float srcy = c[3]*x+c[4]*y+c[5];
      T value = getpixel_bilinear(srcx, srcy, default_value);
      //printf("dst=(%d,%d) --- src = (%f, %f) value=%f\n", x, y, srcx, srcy, value);
      (*result)(x-xmin,y-ymin) = value;
    }
  }

  if (offset_x != 0) *offset_x = xmin;
  if (offset_y != 0) *offset_y = ymin;

  return result;
}

template <typename T>
Matrix<T> * Image<T>::comb_lineal_forward(int sx, int sy, int ancho, int alto, int miniancho, int minialto, LinearCombConf<T> *conf) {

  using april_utils::max;
  using april_utils::min;
  int output_size = miniancho*minialto;

//  printf("Preparing Combination %d\n", conf->patternsize);
  Matrix<T> *mat = new Matrix<T>(1, output_size);
 
  mat->fill(1);

  //FIXME: This is not working correctly if ancho y alto are not multiples of 2
  int miny = max(sy-alto/2,0);
  int maxy = min(sy+alto/2-1, this->height);
  int minx = max(sx-ancho/2,0);
  int maxx = min(sx+ancho/2-1, this->width);
  
  int dx = (sx-ancho/2);
  int dy = (sy-alto/2);

  const float th = 1-1e-5 ;
  int *vec_tuplas = conf->numTuplas + (miny-dy)*ancho + minx-dx;
  for (int y = miny; y < maxy; ++y,vec_tuplas+=ancho){
    for (int x = minx; x < maxx; x++){
      float value = (*this)(x,y);
      if (value < th) {
	value = 1.0-value;
	for (int j = vec_tuplas[x-1];j<vec_tuplas[x];j++) {
	  int dest = conf->indices[j];
	  // FIXME sustituir por mat[dest], ahora no funciona
	  (*mat)(dest) -= value*conf->pesos[j];
	}
      }
    } 
  }
  return mat;
}
template <typename T>
Image<T>* Image<T>::substract_image(Image<T> *img, T low, T high) const {

    int  s_height = img->height;
    int  s_width  = img->width;

    //  printf("%d %d %d %d\n", s_width, s_height, this->width, this->height);
    assert("The images does not have the same dimension"
            && s_width == this->width && s_height == this->height);

    int dims[2];
    dims[0] = this->height;
    dims[1] = this->width;
    Matrix<T> *mat = new Matrix<T>(2,dims, 0);
    Image<T>  *res = new Image<T>(mat);


    for (int y = 0; y < this->height; y++){
        for (int x = 0; x < this->width; x++){
            T value = high + (*this)(x,y) -(*img)(x,y);
            (*res)(x, y) =  april_utils::clamp(value, low, high);
        } 
    }

    return res;
}

#endif // _IMAGE_CC_
