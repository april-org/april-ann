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
#include "buffered_gzfile.h"
#include "buffered_file.h"
#include "clamp.h"
#include "ignore_result.h"
#include <cmath>
#include <cstdio>

using april_utils::clamp;

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

void writeMatrixFloatToFile(MatrixFloat *mat,
			    const char *filename,
			    bool is_ascii) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "w");
    writeMatrixToStream(mat, f, FloatAsciiSizer(), FloatBinarySizer(),
			FloatAsciiCoder<BufferedGZFile>(),
			FloatBinaryCoder<BufferedGZFile>(),
			is_ascii);
  }
  else {
    BufferedFile f(filename, "w");
    writeMatrixToStream(mat, f, FloatAsciiSizer(), FloatBinarySizer(),
			FloatAsciiCoder<BufferedFile>(),
			FloatBinaryCoder<BufferedFile>(),
			is_ascii);
  }
}

char *writeMatrixFloatToString(MatrixFloat *mat,
			       bool is_ascii,
			       int &len) {
  WriteBufferWrapper wrapper;
  len = writeMatrixToStream(mat, wrapper,
			    FloatAsciiSizer(),
			    FloatBinarySizer(),
			    FloatAsciiCoder<WriteBufferWrapper>(),
			    FloatBinaryCoder<WriteBufferWrapper>(),
			    is_ascii);
  return wrapper.getBufferProperty();
}

void writeMatrixFloatToLuaString(MatrixFloat *mat,
				 lua_State *L,
				 bool is_ascii) {
  WriteLuaBufferWrapper wrapper(L);
  IGNORE_RESULT(writeMatrixToStream(mat, wrapper,
				    FloatAsciiSizer(),
				    FloatBinarySizer(),
				    FloatAsciiCoder<WriteLuaBufferWrapper>(),
				    FloatBinaryCoder<WriteLuaBufferWrapper>(),
				    is_ascii));
  wrapper.finish();
}

MatrixFloat *readMatrixFloatFromFile(const char *filename, const char *order) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "r");
    return readMatrixFromStream<BufferedGZFile, float>(f, FloatAsciiExtractor(),
						       FloatBinaryExtractor(),
						       order);
  }
  else {
    BufferedFile f(filename, "r");
    return readMatrixFromStream<BufferedFile, float>(f, FloatAsciiExtractor(),
						     FloatBinaryExtractor(),
						     order);
  }
}

MatrixFloat *readMatrixFloatFromString(constString &cs) {
  return readMatrixFromStream<constString, float>(cs,
						  FloatAsciiExtractor(),
						  FloatBinaryExtractor());
}

//////////////////////////////////////////////////////////////////////////////

void writeMatrixFloatToTabFile(MatrixFloat *mat, const char *filename) {
  if (GZFileWrapper::isGZ(filename)) {
    BufferedGZFile f(filename, "w");
    writeMatrixToTabStream(mat, f, FloatAsciiSizer(),
			   FloatAsciiCoder<BufferedGZFile>());
  }
  else {
    BufferedFile f(filename, "w");
    writeMatrixToTabStream(mat, f, FloatAsciiSizer(),
			   FloatAsciiCoder<BufferedFile>());
  }
}

MatrixFloat *readMatrixFloatFromTabFile(const char *filename,
					const char *order) {
  if (GZFileWrapper::isGZ(filename)) {
    int ncols=0, nrows=0;
    do {
      BufferedGZFile  f(filename, "r");
      FloatAsciiExtractor extractor;
      constString     line("");
      float           value;
      while (f.good()) {
	line = f.extract_u_line();
	if (line.len() > 0) {
	  if (ncols == 0)
	    while(extractor(line,value)) ++ncols;
	  ++nrows;
	}
      }
    } while(false);
    if (nrows <= 0 || ncols <= 0) ERROR_EXIT(256, "Found 0 rows or 0 cols\n");
    BufferedGZFile f(filename, "r");
    return readMatrixFromTabStream<BufferedGZFile, float>(nrows, ncols, f,
							  FloatAsciiExtractor(),
							  order);
  }
  else {
    int ncols=0, nrows=0;
    do {
      BufferedFile    f(filename, "r");
      FloatAsciiExtractor extractor;
      constString     line("");
      float           value;
      while (f.good()) {
	line = f.extract_u_line();
	if (line.len() > 0) {
	  if (ncols == 0)
	    while(extractor(line,value)) ++ncols;
	  ++nrows;
	}
      }
    } while(false);
    if (nrows <= 0 || ncols <= 0) ERROR_EXIT(256, "Found 0 rows or 0 cols\n");
    BufferedFile f(filename, "r");
    return readMatrixFromTabStream<BufferedFile, float>(nrows, ncols, f,
							FloatAsciiExtractor(),
							order);
  }
  return 0;
}

void writeMatrixFloatToTabGZStream(MatrixFloat *mat, BufferedGZFile *stream) {
  writeMatrixToTabStream(mat, *stream, FloatAsciiSizer(),
			 FloatAsciiCoder<BufferedGZFile>());
}

void writeMatrixFloatToTabStream(MatrixFloat *mat, FILE *f) {
  FileWrapper file_wrapper(f);
  BufferedFile stream(file_wrapper);
  writeMatrixToTabStream(mat, stream, FloatAsciiSizer(),
			 FloatAsciiCoder<BufferedFile>());
}
