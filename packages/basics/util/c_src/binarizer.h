/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef BINARIZER_H
#define BINARIZER_H

extern "C" {
#include <stdint.h>
}

namespace april_utils {

  // glibc defines these ones in string.h 
  // they're gone in C99, but stay in C++ :_(
#undef BIG_ENDIAN
#undef LITTLE_ENDIAN

  // clase con métodos estáticos:
  class binarizer {
  
    static char cod[]; // tablas
    static char uncod[];

    static enum endian_enum {BIG_ENDIAN, LITTLE_ENDIAN, UNKNOWN_ENDIAN} endianness;

  public:
    static void init();
  
    static void code_uint32(uint32_t i, char b[5]);
    static void code_int32 ( int32_t i, char b[5]);
    static uint32_t decode_uint32(const char b[5]);
    static int32_t  decode_int32 (const char b[5]);
  
    static void code_float ( float i, char b[5]);
    static void code_double(double i, char b[10]);
    static float  decode_float (const char b[5]);
    static double decode_double(const char b[10]);

#ifdef HAVE_UINT16
    static void code_uint16(uint16_t i, char b[3]);
    static void code_int16 ( int16_t i, char b[3]);
    static uint16_t decode_uint16(const char b[3]);
    static int16_t  decode_int16 (const char b[3]);
#endif

#ifdef HAVE_UINT64
    static void code_uint64(uint64_t i, char b[10]);
    static void code_int64 ( int64_t i, char b[10]);
    static uint64_t decode_uint64(const char b[10]);
    static int64_t  decode_int64 (const char b[10]);
#endif


    // estima (por exceso) la longitud en bytes al escribir n numeros de
    // XXX bits con la opción de poner '\n' cada 80 caracteres aprox:

    static int buffer_size_16(int num, bool with_newlines=true);
    static int buffer_size_32(int num, bool with_newlines=true);
    static int buffer_size_64(int num, bool with_newlines=true);

    // El objetivo es codificar el vector y dejar el resultado en el
    // buffer. Devuelve el numero de caracteres utilizados en el
    // buffer. Si el buffer no es lo bastante grande para guardar todo
    // el contenido, no hace nada y devuelve 0 caracteres utilizados.
    static int code_vector_float(const float *vect, int vect_size,
                                 char *dest_buffer, int dest_buffer_size,
                                 bool with_newlines=true);

    template<typename ITERATOR>
    static int code_iterator_float(ITERATOR it, ITERATOR end, int vect_size,
                                   char *dest_buffer, int dest_buffer_size,
                                   bool with_newlines=true) {
      int i;
      char *rb = dest_buffer;
      if (buffer_size_32(vect_size,with_newlines) <= dest_buffer_size) {
        i = 0;
        while (it != end) {
          code_float(*it,rb);
          rb+=5;
          i++;
          ++it;
          if (with_newlines && !(i%16)) { *rb = '\n'; rb++; }
        }
        if (with_newlines && (i % 16))  { *rb = '\n'; rb++; }
        *rb = '\0'; rb++;
      }
      return rb-dest_buffer;
    }
  
    static int code_vector_int32(const int32_t *vect, int vect_size,
                                 char *dest_buffer, int dest_buffer_size,
                                 bool with_newlines=true);
  
  };

} // namespace april_utils

#endif // BINARIZER_H
