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
#include <png.h>

#include "clamp.h"
#include "libpng.h"

namespace Imaging {
  namespace LibPNG {
    ImageFloatRGB *readPNG(const char *filename)
    {
      png_structp png_ptr;
      png_infop   info_ptr;

      png_uint_32 width, height;
      int bit_depth, color_type;
      unsigned char *image_data;

      // Open input file
      FILE *fp = fopen(filename, "rb");

      if (!fp) {
        fprintf(stderr, "LibPNG::readPNG() -> cannot open input file %s\n", filename);
        return NULL;
      }

      // Check signature
      unsigned char sig[8];
      if (fread(sig, 1, 8, fp) < 8) {
        fprintf(stderr, "LibPNG::readPNG() -> cannot read signature from file %s\n", filename);
        return NULL;
      }

      if (!png_check_sig(sig, 8)) {
        fprintf(stderr, "LibPNG::readPNG() -> %s is not a PNG file\n", filename);
        return NULL;
      }

      // Create read and info structs
      png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (!png_ptr) {
        fprintf(stderr, "LibPNG::readPNG() -> cannot allocate memory for png_struct\n");
        return NULL;
      }

      info_ptr = png_create_info_struct(png_ptr);
      if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fprintf(stderr, "LibPNG::readPNG() -> cannot allocate memory for png_struct\n");
        return NULL;
      }

      // Set error handling and init I/O
      if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fprintf(stderr, "Error while initializing I/O\n");
        return NULL;
      }

      png_init_io(png_ptr, fp);
      png_set_sig_bytes(png_ptr, 8); // 8 bytes already skipped

      // Read PNG info ; get image attributes
      png_read_info(png_ptr, info_ptr);
      png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth,
                   &color_type, NULL, NULL, NULL);

      // ignore alpha
      png_set_strip_alpha(png_ptr);

      // Expand palette to full RGB
      if (color_type == PNG_COLOR_TYPE_PALETTE) 
        png_set_palette_to_rgb(png_ptr);

      // Expand low-bit-depth grayscale to 8 bit
      if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) 
        png_set_expand_gray_1_2_4_to_8(png_ptr);

      // Expand grayscale to RGB
      if (color_type == PNG_COLOR_TYPE_GRAY)
        png_set_gray_to_rgb(png_ptr);

      // Reduce depth to 8 bits/channel if needed
      if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    
      // Prepare to read pixels
      if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fprintf(stderr, "Error while reading PNG image from %s\n", filename);
        return NULL;
      }

      png_uint_32 rowbytes;
      int channels;

      png_read_update_info(png_ptr, info_ptr);
      rowbytes = png_get_rowbytes(png_ptr, info_ptr);
      channels = (int)png_get_channels(png_ptr, info_ptr);

      image_data = new unsigned char[rowbytes*height];

      // libpng needs an array with pointers to the beginning of each row
      png_bytep *row_pointers = new png_bytep[height];
      for (unsigned int i=0; i<height; i++)
        row_pointers[i] = image_data + i*rowbytes;

      png_read_image(png_ptr, row_pointers);

      // Copy read data to a ImageFloatRGB
      int dims[2]={static_cast<int>(height),
                   static_cast<int>(width)};
      Basics::Matrix<FloatRGB> *m = new Basics::Matrix<FloatRGB>(2, dims);
      AprilMath::MatrixExt::Initializers::matFill(m, FloatRGB(0.0f,0.0f,0.0f));
      ImageFloatRGB *res = new ImageFloatRGB(m);

      unsigned char *p = image_data;
      for (unsigned int y=0; y<height; y++) {
        for (unsigned int x=0; x<width; x++) {
          float r = *p/255.0f;
          float g = *(p+1)/255.0f;
          float b = *(p+2)/255.0f;
          p+=3;
          (*res)(x,y) = FloatRGB(r,g,b);
        }
      }

      // cleanup
      delete[] row_pointers;
      delete[] image_data;
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
      fclose(fp);

      return res;
    }

    bool writePNG(ImageFloatRGB *img, const char *filename)
    {
      bool success=false; // set to true when all ends correctly
      png_structp png_ptr;
      png_infop   info_ptr;
      unsigned char *image_data=NULL, *p=NULL;
      png_bytep *row_pointers = NULL;
      png_uint_32 rowbytes = 3*img->width();

      // Open file for writing
      FILE *fp = fopen(filename, "wb");
      if (!fp) {
        fprintf(stderr, "LibPNG::writePNG() -> cannot open output file\n");
        return false;
      }
    
      // Create png struct and info struct
      png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (!png_ptr) {
        fprintf(stderr, "LibPNG::writePNG() -> cannot allocate memory for png_struct\n");
        goto close_file;
      }

      info_ptr = png_create_info_struct(png_ptr);
      if (!info_ptr) {
        fprintf(stderr, "LibPNG::writePNG() -> cannot allocate memory for png_struct\n");
        goto destroy_png_struct;
      }

      // Init I/O
      if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "LibPNG::writePNG() -> Error while initializing I/O\n");
        goto destroy_png_struct;
      }

      png_init_io(png_ptr, fp);

      // Write header (fixed to RGB, 24bpp)
      if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "LibPNG::writePNG() -> Error while writing header\n");
        goto destroy_png_struct;
      }

      png_set_IHDR(png_ptr, info_ptr, img->width(), img->height(), 8,
                   PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                   PNG_FILTER_TYPE_BASE);

      png_write_info(png_ptr, info_ptr);

      // Create aux pixel buffer
      image_data = new unsigned char[img->width() * img->height() * 3];
      p = image_data; 
      for (int y=0; y < img->height(); y++) {
        for (int x=0; x < img->width(); x++) {
          FloatRGB c = (*img)(x,y);
          *p++ = (unsigned char)(AprilUtils::clamp(c.r, 0.0f, 1.0f)*255.0f);
          *p++ = (unsigned char)(AprilUtils::clamp(c.g, 0.0f, 1.0f)*255.0f);
          *p++ = (unsigned char)(AprilUtils::clamp(c.b, 0.0f, 1.0f)*255.0f);
        }
      }
        
      // Create row_pointers array
      row_pointers = new png_bytep[img->height()];
      for (int i=0; i < img->height(); i++)
        row_pointers[i] = image_data + i*rowbytes;

      // Write pixel data
      if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "LibPNG::writePNG() -> Error while writing pixel data\n");
        goto delete_aux_buffers;
      }

      png_write_image(png_ptr, row_pointers);

      // End write
      if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "LibPNG::writePNG() -> Error while ending write\n");
        goto delete_aux_buffers;
      }

      png_write_end(png_ptr, NULL);

      success = true;

      // Cleanup (various stages)
    delete_aux_buffers:
      delete[] row_pointers;
      delete[] image_data;

    destroy_png_struct:
      png_destroy_write_struct(&png_ptr, &info_ptr);
    close_file:
      fclose(fp);
      return success;
    }

  } // namespace LibPNG
} // namespace Imaging


