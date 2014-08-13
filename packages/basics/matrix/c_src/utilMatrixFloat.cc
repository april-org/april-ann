/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
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
#include "utilMatrixFloat.h"
#include "binarizer.h"
#include "clamp.h"
#include "ignore_result.h"
#include <cmath>
#include <cstdio>

using april_utils::clamp;
using april_utils::constString;
using april_io::StreamInterface;

namespace basics {
  
  /////////////////////////////////////////////////////////////////////////
  
  template<>
  bool AsciiExtractor<float>::operator()(constString &line,
                                         float &destination) {
    bool result = line.extract_float(&destination);
    if (!result) return false;
    return true;
  }
  
  template<>
  bool BinaryExtractor<float>::operator()(constString &line,
                                          float &destination) {
    if (!line.extract_float_binary(&destination)) return false;
    return true;
  }
  
  template<>
  int AsciiSizer<float>::operator()(const Matrix<float> *mat) {
    return mat->size()*12;
  }

  template<>
  int BinarySizer<float>::operator()(const Matrix<float> *mat) {
    return april_utils::binarizer::buffer_size_32(mat->size());
  }

  template<>
  void AsciiCoder<float>::operator()(const float &value,
                                     april_io::StreamInterface *stream) {
    stream->printf("%.5g", value);
  }
  
  template<>
  void BinaryCoder<float>::operator()(const float &value,
                                      april_io::StreamInterface *stream) {
    char b[5];
    april_utils::binarizer::code_float(value, b);
    stream->put(b, sizeof(char)*5);
  }

  /////////////////////////////////////////////////////////////////////////////
  
  template<>
  int SparseAsciiSizer<float>::operator()(const SparseMatrix<float> *mat) {
    return mat->nonZeroSize()*12;
  }
  
  template<>
  int SparseBinarySizer<float>::operator()(const SparseMatrix<float> *mat) {
    return april_utils::binarizer::buffer_size_32(mat->nonZeroSize());
  }
  
  //////////////////////////////////////////////////////////////////////////

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
    MatrixFloat::iterator it(mat->begin());
    int size2 = 2*width*height;
    for (int i=0; i<size2; i+=2) {
      float brillo = (hexdigit(cs[i])*16+hexdigit(cs[i+1])) * normaliza;
      *it = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
      ++it;
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
      MatrixFloat::iterator it(mat->begin());
      const char *b = cs;
      int k=0;
      int vector_pos = 0;
      for (int row=0; row<dims[1]; ++row) {
        int vector_offset = 0;
        for (int col=0; col<dims[0]; ++col, ++k, ++it) {
          float brillo = (float)(((unsigned char)b[vector_pos] >> (7-vector_offset)) & 0x01);	
          *it = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
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
        MatrixFloat::iterator it(mat->begin());
        const char *b = cs;
        for (unsigned int i=0;i<npixels;i++) {
          float v = clamp(normaliza*(unsigned char)b[i],CTEBLANCO,CTENEGRO);
          *it = v; ++it;
          *it = v; ++it;
          *it = v; ++it;
        }
      } else {
        mat = new MatrixFloat(2,dims);
        MatrixFloat::iterator it(mat->begin());
        const char *b = cs;
        for (unsigned int i=0;i<npixels;i++,++it) {
          float brillo = normaliza*(unsigned char)b[i];
          *it = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
        }
      }
    } else if (formato == "P6") { // PPM BINARY
      if (!linea.extract_int(&maxval)) return 0;
      float normaliza = 1.0/maxval;
      if (cs.len() < 3*npixels) return 0;
      if (forcegray) {
        mat = new MatrixFloat(2,dims);
        MatrixFloat::iterator it(mat->begin());
        const char *b = cs;
        // gray = .3 * red + .59 * green + .11 * blue
        normaliza *= 0.01;
        for (unsigned int i=0;i<npixels;i++,++it) {
          float brillo = normaliza*((unsigned char)b[0]*30+
                                    (unsigned char)b[1]*59+
                                    (unsigned char)b[2]*11);
          *it = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
          b += 3;
        }
      } else {
        mat = new MatrixFloat(3,dims);
        MatrixFloat::iterator it(mat->begin());
        const char *b = cs;
        // This loop does not work because in both cases we have 3-tuples
        // with values of the same pixel
        npixels *= 3;
        for (unsigned int i=0;i<npixels;i++,++it) {
          float brillo = normaliza*(unsigned char)b[i];
          *it = clamp(CTENEGRO + (CTEBLANCO-CTENEGRO)*brillo,CTEBLANCO,CTENEGRO);
        }
      }  
    } else {
      // Non-recognised format, nothing to be done
    }
    return mat;
  }

  int saveMatrixFloatPNM(MatrixFloat *mat,
                         char **buffer) {
    if ((mat->getNumDim() < 2) || (mat->getNumDim() > 3)) {
      *buffer = 0; return 0;
    }
    int i,ancho,alto,prof,sizedata,sizeheader = 100;
    ancho = mat->getDimSize(1);
    alto  = mat->getDimSize(0);
    prof = (mat->getNumDim() == 3) ? mat->getDimSize(2) : 1;
    if (prof != 1 && prof != 3) {
      *buffer = 0; return 0;
    }
    sizedata = ancho*alto*prof;
    char *r,*b;
    r = b = new char[sizedata+sizeheader];
    r += sprintf(r,"%s\n%d %d\n255\n",
                 ((prof == 1) ? "P5" : "P6"),ancho,alto);
    float f;
    MatrixFloat::const_iterator it(mat->begin());
    for (i=0;i<sizedata;i++,++it) {
      f = clamp((CTENEGRO - (*it)) * 1/(CTENEGRO-CTEBLANCO),CTEBLANCO,CTENEGRO);
      r[i] = clamp((unsigned char)roundf(f*255),(unsigned char)0,(unsigned char)255);
    }
    *buffer = b;
    return sizedata+(r-b);
  }

  int saveMatrixFloatHEX(MatrixFloat *mat,
                         char **buffer,
                         int *width, int *height) {
    if (mat->getNumDim() != 2) {
      *buffer = 0; return 0;
    }
    int i,ancho,alto,sizedata,sizedata2;
    ancho = mat->getDimSize(1);
    alto  = mat->getDimSize(0);
    sizedata   = ancho*alto;
    sizedata2  = 2*ancho*alto;
    char *r,*b;
    r = b = new char[sizedata2+1];
    float f;
    MatrixFloat::const_iterator it(mat->begin());
    for (i=0;i<sizedata2;i+=2, ++it) {
      f = clamp((CTENEGRO - (*it)) * 1/(CTENEGRO-CTEBLANCO),CTEBLANCO,CTENEGRO);
      unsigned int aux = clamp((unsigned int)roundf(f*255),(unsigned int)0,(unsigned int)255);
      sprintf(&r[i], "%02x", aux);
    }
    *buffer = b;
    *width  = ancho;
    *height = alto;
    return sizedata2+(r-b);
  }
  
} // namespace basics
