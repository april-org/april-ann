/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera
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
#include "read_file_stream.h"
#include "utilMatrixFloat.h"
#include "binarizer.h"
#include "clamp.h"
#include <cmath>
#include <cstdio>

using april_utils::clamp;

template <typename T>
MatrixFloat* readMatrixFloatFromStream(T &stream) {
  constString linea,formato,token;
  // First we read the matrix dimensions
  linea = stream.extract_u_line();
  if (!linea) return 0;
  static const int maxdim=100;
  int dims[maxdim];
  int n=0, pos_comodin=-1;
  while (n<maxdim && (token = linea.extract_token())) {
    if (token == "*") {
      if (pos_comodin != -1) {
	// Error, more than one comodin
	fprintf(stderr,"more than one '*' reading a matrix\n");
	return 0;
      }
      pos_comodin = n;
    } else if (!token.extract_int(&dims[n])) {
      return 0;
    }
    n++;
  }
  if (n==maxdim) return 0; // Maximum allocation problem
  MatrixFloat *mat = 0;
  // Now we read the type of the format
  linea = stream.extract_u_line();
  formato = linea.extract_token();
  if (!formato) {delete mat; return 0;}
  if (pos_comodin == -1) { // Normal version
    mat = new MatrixFloat(n,dims);
    int i=0;
    if (formato == "ascii") {
      while (i<mat->size && (linea=stream.extract_u_line()))
	while (i<mat->size && linea.extract_float(&mat->data->get(i))) { i++; }
    } else { // binary
      while (i<mat->size && (linea=stream.extract_u_line()))
	while (i<mat->size && linea.extract_float_binary(&mat->data->get(i))) { i++; }
    }
    if (i != mat->size) { delete mat; mat = 0; }
  } else { // version with comodin
    int size=0,maxsize=4096;
    float *data = new float[maxsize];
    if (formato == "ascii") {
      while ( (linea=stream.extract_u_line()) )
	while (linea.extract_float(&data[size])) { 
	  size++; 
	  if (size == maxsize) { // resize data vector
	    float *aux = new float[2*maxsize];
	    for (int a=0;a<maxsize;a++)
	      aux[a] = data[a];
	    maxsize *= 2;
	    delete[] data; data = aux;
	  }
	}
    } else { // binary
      while ( (linea=stream.extract_u_line()) )
	while (linea.extract_float_binary(&data[size])) { 
	  size++; 
	  if (size == maxsize) { // resize data vector
	    float *aux = new float[2*maxsize];
	    for (int a=0;a<maxsize;a++)
	      aux[a] = data[a];
	    maxsize *= 2;
	    delete[] data; data = aux;
	  }
	}
    }
    int sizesincomodin = 1;
    for (int i=0;i<n;i++)
      if (i != pos_comodin)
	sizesincomodin *= dims[i];
    if ((size % sizesincomodin) != 0) {
      // Error: The size of the data does not coincide
      fprintf(stderr,"data size is not valid reading a matrix with '*'\n");
      delete[] data; return 0;
    }
    dims[pos_comodin] = size / sizesincomodin;
    mat = new MatrixFloat(n,dims);
    for (int i=0; i<size; i++) mat->data->get(i) = data[i];
    delete[] data;
  }
  return mat;
}

// Returns the string length (there is a '\0' that is not counted)
void saveMatrixFloatToFile(MatrixFloat *mat, FILE *f, bool is_ascii) {
  const int columns = 9;
  int i;
  for (i=0;i<mat->numDim-1;i++)
    fprintf(f, "%d ",mat->matrixSize[i]);
  fprintf(f,"%d\n",mat->matrixSize[mat->numDim-1]);
  if (is_ascii) {
    fprintf(f,"ascii\n");
    for (i=0;i<mat->size;i++) {
      fprintf(f,"%.5g%c",mat->data->get(i),
	      ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      fprintf(f,"\n"); 
    }
  } else { // binary
    fprintf(f,"binary\n");
    // We substract 1 so the final '\0' is not considered
    char b[5];
    for (i=0; i<mat->size; ++i) {
      binarizer::code_float(mat->data->get(i), b);
      fprintf(f, "%c%c%c%c%c%c", b[0], b[1], b[2], b[3], b[4], 
	      ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0)
      fprintf(f,"\n"); 
  }
}

// Returns the string length (there is a '\0' that is not counted)
int saveMatrixFloatToString(MatrixFloat *mat, char **buffer, bool is_ascii) {
  const int columns = 9;
  int i,sizedata,sizeheader;
  sizeheader = mat->numDim*10+10; // FIXME: To put adequate values
  if (is_ascii)
    sizedata = mat->size*12; // Memory used by float in ascii
			     // including spaces, enters, etc...
  else
    sizedata = binarizer::buffer_size_32(mat->size);
  char *r, *b;
  r = b = new char[sizedata+sizeheader];
  for (i=0;i<mat->numDim-1;i++)
    r += sprintf(r,"%d ",mat->matrixSize[i]);
  r += sprintf(r,"%d\n",mat->matrixSize[mat->numDim-1]);
  if (is_ascii) {
    r += sprintf(r,"ascii\n");
    for (i=0;i<mat->size;i++) {
      r += sprintf(r,"%.5g%c",mat->data->get(i),
		   ((((i+1) % columns) == 0) ? '\n' : ' '));
    }
    if ((i % columns) != 0) {
      r += sprintf(r,"\n"); 
    }
  } else { // binary
    r += sprintf(r,"binary\n");
    // We substract 1 so the final '\0' is not considered
    r += -1 + binarizer::code_vector_float(mat->getData(),
					   mat->size,
					   r, sizedata);
  }
  *buffer = b;
  return r-b;
}

inline int hexdigit(char c) {
  if ('0'<=c && c<='9') return c-'0';
  if ('A'<=c && c<='F') return c-'A'+10;
  else                  return c-'a'+10;  
}

MatrixFloat* readMatrixFloatHEX(int width,
				int height, 
				constString cs) {
  float normaliza = 1.0/255;
  int dims[2]; // dims[0] is height, dims[1] is width
  dims[1] = width;
  dims[0] = height;
  MatrixFloat *mat = new MatrixFloat(2,dims);
  float *d = mat->getData();
  int size2 = 2*width*height;
  for (int i=0; i<size2; i+=2) {
    float brillo = (hexdigit(cs[i])*16+hexdigit(cs[i+1])) * normaliza;
    d[i>>1] = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
  }
  return mat;
}
  
MatrixFloat* readMatrixFloatPNM(constString cs, 
				bool forcecolor, 
				bool forcegray) {
  // file format specification:
  // PBM_TEXT P1
  // PGM_TEXT P2
  // PPM_TEXT P3
  // PBM_BIN  P4
  // PGM_BIN  P5
  // PPM_BIN  P6
  constString linea,formato;
  // We read the format of the image
  linea = cs.extract_u_line();
  formato = linea.extract_token();
  if (!formato) return 0;
  // We read the image dimensions
  linea = cs.extract_u_line();
  int dims[3],maxval=1;
  unsigned int npixels; // dims[0] is height, dims[1] is width
  if (!linea.extract_int(&dims[1])) return 0;
  if (!linea.extract_int(&dims[0])) return 0;
  dims[2] = 3; // In case of using this field, it will always be 3
  npixels = dims[1]*dims[0];
  // We read the maximum value
  linea = cs.extract_u_line();
  cs.skip(1); // IMPORTANT: Read the '\n' later
  MatrixFloat *mat = 0;
  // From here, it depends on the format
  if (formato == "P4") { // PBM BINARY
    //if (cs.len() < (npixels >> 3)) return 0;
    mat = new MatrixFloat(2,dims);
    float *d = mat->getData();
    const char *b = cs;
    int k=0;
    int vector_pos = 0;
    for (int row=0; row<dims[1]; ++row) {
      int vector_offset = 0;
      for (int col=0; col<dims[0]; ++col, ++k) {
	float brillo = (float)(((unsigned char)b[vector_pos] >> (7-vector_offset)) & 0x01);	
	d[k] = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
	if (vector_offset == 7) {
	  if (col+1 != dims[0])
	    vector_pos = vector_pos + 1;
	}
	vector_offset = (vector_offset + 1) % 8;
      }
      vector_pos = vector_pos + 1;
    }
  } else if (formato == "P5") { // PGM BINARY
    if (!linea.extract_int(&maxval)) return 0;
    float normaliza = 1.0/maxval;
    if (cs.len() < npixels) return 0;
    if (forcecolor) { 
      // We make the 3 values to be equal
      mat = new MatrixFloat(3,dims);
      float *d = mat->getData();
      const char *b = cs;
      for (unsigned int i=0;i<npixels;i++) {
	d[0] = d[1] = d[2] = clamp(normaliza*(unsigned char)b[i],CTEBLANCO,CTENEGRO);
	d += 3;
      }
    } else {
      mat = new MatrixFloat(2,dims);
      float *d = mat->getData();
      const char *b = cs;
      for (unsigned int i=0;i<npixels;i++) {
	float brillo = normaliza*(unsigned char)b[i];
	d[i] = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
      }
    }
  } else if (formato == "P6") { // PPM BINARY
    if (!linea.extract_int(&maxval)) return 0;
    float normaliza = 1.0/maxval;
    if (cs.len() < 3*npixels) return 0;
    if (forcegray) {
      mat = new MatrixFloat(2,dims);
      float *d = mat->getData();
      const char *b = cs;
      // gray = .3 * red + .59 * green + .11 * blue
      normaliza *= 0.01;
      for (unsigned int i=0;i<npixels;i++) {
	float brillo = normaliza*((unsigned char)b[0]*30+
				  (unsigned char)b[1]*59+
				  (unsigned char)b[2]*11);
	d[i] = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
	b += 3;
      }
    } else {
      mat = new MatrixFloat(3,dims);
      float *d = mat->getData();
      const char *b = cs;
      // This loop does not work because in both cases we have 3-tuples
      // with values of the same pixel
      npixels *= 3;
      for (unsigned int i=0;i<npixels;i++) {
	float brillo = normaliza*(unsigned char)b[i];
	d[i] = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
      }
    }  
  } else {
    // Non-recognised format, nothing to be done
  }
  return mat;
}

int saveMatrixFloatPNM(MatrixFloat *mat,
		       char **buffer) {
  if ((mat->numDim < 2) || (mat->numDim > 3)) {
    *buffer = 0; return 0;
  }
  int i,ancho,alto,prof,sizedata,sizeheader = 100;
  ancho = mat->matrixSize[1];
  alto  = mat->matrixSize[0];
  prof = (mat->numDim == 3) ? mat->matrixSize[2] : 1;
  if (prof != 1 && prof != 3) {
    *buffer = 0; return 0;
  }
  sizedata = ancho*alto*prof;
  char *r,*b;
  r = b = new char[sizedata+sizeheader];
  r += sprintf(r,"%s\n%d %d\n255\n",
	       ((prof == 1) ? "P5" : "P6"),ancho,alto);
  float f,*v = mat->getData();
  for (i=0;i<sizedata;i++) {
    f = clamp((CTENEGRO - v[i]) * 1/(CTENEGRO-CTEBLANCO),CTEBLANCO,CTENEGRO);
    r[i] = clamp((unsigned char)roundf(f*255),(unsigned char)0,(unsigned char)255);
  }
  *buffer = b;
  return sizedata+(r-b);
}

int saveMatrixFloatHEX(MatrixFloat *mat,
		       char **buffer,
		       int *width, int *height) {
  if (mat->numDim != 2) {
    *buffer = 0; return 0;
  }
  int i,ancho,alto,sizedata,sizedata2;
  ancho = mat->matrixSize[1];
  alto  = mat->matrixSize[0];
  sizedata   = ancho*alto;
  sizedata2  = 2*ancho*alto;
  char *r,*b;
  r = b = new char[sizedata2+1];
  float f,*v = mat->getData();
  for (i=0;i<sizedata2;i+=2) {
    f = clamp((CTENEGRO - v[i>>1]) * 1/(CTENEGRO-CTEBLANCO),CTEBLANCO,CTENEGRO);
    unsigned int aux = clamp((unsigned int)roundf(f*255),(unsigned int)0,(unsigned int)255);
    sprintf(&r[i], "%02x", aux);
  }
  *buffer = b;
  *width  = ancho;
  *height = alto;
  return sizedata2+(r-b);
}

template MatrixFloat *readMatrixFloatFromStream<constString>(constString &stream);
template MatrixFloat *readMatrixFloatFromStream<ReadFileStream>(ReadFileStream &stream);
