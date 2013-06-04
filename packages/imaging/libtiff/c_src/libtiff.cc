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
#include <cstdlib>
#include <stdint.h>
#include "tiffio.h"

#include "clamp.h"
#include "libtiff.h"

namespace LibTIFF
{
  ImageFloatRGB *readTIFF(const char *filename)
  {
    TIFF* tif = TIFFOpen(filename, "r");
    if (tif == NULL) {
      fprintf(stderr, "LibTIFF::readTIFF() -> TIFFOpen failed on %s\n", filename);
      return NULL;
    }

    int width, height;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    
    int dims[2]={height, width};
    int ok;
    Matrix<FloatRGB> *m = NULL;
    ImageFloatRGB *res = NULL;
    uint32_t *p = NULL;
    
    int npixels = width * height;
    uint32_t *raster = (uint32_t*) _TIFFmalloc(npixels * sizeof (uint32_t));
    if (raster == NULL) {
      fprintf(stderr, "LibTIFF::readTIFF() -> _TIFFMalloc failed on %s\n", filename);
      goto close_file;
    }

    ok = TIFFReadRGBAImage(tif, width, height, raster, 0);
    if (!ok) {
      fprintf(stderr, "LibTIFF::readTIFF() -> TIFFReadRGBAImage failed on %s\n", filename);
      goto free_image_data;
    }

    m   = new Matrix<FloatRGB>(2, dims, FloatRGB(0.0f,0.0f,0.0f));
    res = new ImageFloatRGB(m);
    /* process raster data */
    p = raster;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        uint32_t pixel = *p++;
        float r = TIFFGetR(pixel)/255.0f;
        float g = TIFFGetG(pixel)/255.0f;
        float b = TIFFGetB(pixel)/255.0f;
        (*res)(x,height-1-y) = FloatRGB(r,g,b);
      }
    }
    
    free_image_data:
    _TIFFfree(raster);
    
    close_file:
    TIFFClose(tif);

    return res;
  }

  bool writeTIFF(ImageFloatRGB *img, const char *filename)
  {
    bool result = false;
    TIFF* tif = TIFFOpen(filename, "w");
    if (tif == NULL) {
      fprintf(stderr, "LibTIFF::writeTIFF() -> TIFFOpen failed on %s\n", filename);
      return result;
    }


    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img->width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img->height);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(tif, TIFFTAG_ARTIST, "");

    int size = (img->width * img->height * 3);
    int i;
    uint8_t* buffer;

    buffer = (uint8_t*)_TIFFmalloc((unsigned int)size);
    if (buffer == NULL) {
      fprintf(stderr, "LibTIFF::writeTIFF() -> _TIFFMalloc failed on %s\n", filename);
      goto close_file;
    }

    // copy data into buffer
    i=0;
    for (int y = 0; y < img->height; y++) {
      for (int x = 0; x < img->width; x++) {
        FloatRGB color((*img)(x,y));
        buffer[i]   = uint8_t(color.r*255.0f);
        buffer[i+1] = uint8_t(color.g*255.0f);
        buffer[i+2] = uint8_t(color.b*255.0f);
        i+=3;
      }
    }
        
    // Write the information to the file
    TIFFWriteEncodedStrip(tif, 0, buffer, size);

    // If we are here, file is OK
    result=true;

    _TIFFfree(buffer);

    close_file:    
    TIFFClose(tif);

    return result;
  }

}



