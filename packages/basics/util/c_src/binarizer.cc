/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include "binarizer.h"

// glibc defines these ones in string.h 
// they're gone in C99, but stay in C++ :_(
#undef BIG_ENDIAN
#undef LITTLE_ENDIAN

// from BASE5 on we need to use long constants to avoid overflow
static const uint32_t BASE  = 85;
static const uint32_t BASE1 = 85;
static const uint32_t BASE2 = 85*85;
static const uint32_t BASE3 = 85*85*85;
static const uint32_t BASE4 = 85*85*85*85;
static const uint64_t BASE5 = uint64_t(85)*85*85*85*85;
static const uint64_t BASE6 = uint64_t(85)*85*85*85*85*85;
static const uint64_t BASE7 = uint64_t(85)*85*85*85*85*85*85;
static const uint64_t BASE8 = uint64_t(85)*85*85*85*85*85*85*85;
static const uint64_t BASE9 = uint64_t(85)*85*85*85*85*85*85*85*85;

binarizer::endian_enum binarizer::endianness;

// vector de talla BASE=85, da el codigo ascii no extendido asociado a
// cada uno de los 85 valores diferentes:
char binarizer::cod[] = {'\041','\044','\045','\046','\047','\050','\051','\052','\053','\054',
			 '\056','\057','\060','\061','\062','\063','\064','\065','\066','\067',
			 '\070','\071','\072','\073','\074','\075','\076','\077','\100','\101',
			 '\102','\103','\104','\105','\106','\107','\110','\111','\112','\113',
			 '\114','\115','\116','\117','\120','\121','\122','\123','\124','\125',
			 '\126','\127','\130','\131','\132','\136','\137','\140','\141','\142',
			 '\143','\144','\145','\146','\147','\150','\151','\152','\153','\154',
			 '\155','\156','\157','\160','\161','\162','\163','\164','\165','\166',
			 '\167','\170','\171','\172','\173',};

// vector de talla 256, dado un valor ascii devuelve el digito en base 85 asociado:
char binarizer::uncod[] = {'\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\001','\002','\003','\004',
			 '\005','\006','\007','\010','\011','\000','\012','\013','\014','\015',
			 '\016','\017','\020','\021','\022','\023','\024','\025','\026','\027',
			 '\030','\031','\032','\033','\034','\035','\036','\037','\040','\041',
			 '\042','\043','\044','\045','\046','\047','\050','\051','\052','\053',
			 '\054','\055','\056','\057','\060','\061','\062','\063','\064','\065',
			 '\066','\000','\000','\000','\067','\070','\071','\072','\073','\074',
			 '\075','\076','\077','\100','\101','\102','\103','\104','\105','\106',
			 '\107','\110','\111','\112','\113','\114','\115','\116','\117','\120',
			 '\121','\122','\123','\124','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000','\000','\000','\000','\000',
			 '\000','\000','\000','\000','\000','\000',};

// initialization, sanity checks, etc
void binarizer::init() {
  double d = 1234.5678; // 40 93 4a 45 6d 5c fa ad
  unsigned char c[8];
  unsigned char big[8] = {0x40,0x93,0x4a,0x45,0x6d,0x5c,0xfa,0xad};
  unsigned char little[8] = {0xad,0xfa,0x5c,0x6d,0x45,0x4a,0x93,0x40};

  memcpy(c, &d, sizeof(double));
  // Check for big endian
  endianness = BIG_ENDIAN;
  for (int i=0; i<8; i++) {
    if (c[i] != big[i]) {
      endianness = UNKNOWN_ENDIAN;
      break;
    }
  }

  if (endianness == UNKNOWN_ENDIAN) {
    // Check for little endian
    endianness = LITTLE_ENDIAN;
    for (int i=0; i<8; i++) {
      if (c[i] != little[i]) {
        endianness = UNKNOWN_ENDIAN;
        break;
      }
    }
  }

  if (endianness == UNKNOWN_ENDIAN) {
    fprintf(stderr, "FATAL ERROR: Unknown endianness or unknown (no IEEE 754) floating-point format\n");
    exit(EXIT_FAILURE);
  }
}


// ------------------- 16 bits -------------------
#ifdef HAVE_UINT16
uint16_t binarizer::decode_uint16(const char b[3]) {
  return
    uncod[(int)b[0]] * BASE2 +
    uncod[(int)b[1]] * BASE1 +
    uncod[(int)b[2]];
}
void binarizer::code_uint16(uint16_t a0, char b[3]) {
  uint16_t a1,a2;
  a1   = a0/BASE;
  a2   = a0/BASE2;
  b[0] = cod[a2];  
  b[1] = cod[a1-a2*BASE];
  b[2] = cod[a0-a1*BASE];
}

// signed case is reduced to unsigned case. Two complement
// representation is assumed
int16_t binarizer::decode_int16(const char b[3]) {
  uint16_t ui;
  int16_t i;
  ui = decode_uint16(b);
  memcpy(&i, &ui, sizeof(uint16_t));
  return i;
}

void binarizer::code_int16(int16_t a0, char b[3]) {
  uint16_t ui;
  memcpy(&ui, &a0, sizeof(uint16_t));
  code_uint16(ui,b);
}
#endif

// ------------------- 32 bits -------------------
uint32_t binarizer::decode_uint32(const char c[5]) {
  return
    uncod[(int)c[0]] * BASE4 +
    uncod[(int)c[1]] * BASE3 +
    uncod[(int)c[2]] * BASE2 +
    uncod[(int)c[3]] * BASE1 +
    uncod[(int)c[4]];
}
void binarizer::code_uint32(uint32_t a0, char b[5]) {
  uint32_t a1,a2,a3,a4;
  a1   = a0/BASE1;
  a2   = a0/BASE2;
  a3   = a0/BASE3;
  a4   = a0/BASE4;
  b[0] = cod[a4];
  b[1] = cod[a3-a4*BASE];
  b[2] = cod[a2-a3*BASE];
  b[3] = cod[a1-a2*BASE];
  b[4] = cod[a0-a1*BASE];
}

// signed case is reduced to unsigned case. Two complement
// representation is assumed
int32_t binarizer::decode_int32(const char b[5]) {
  uint32_t ui;
  int32_t i;
  ui = decode_uint32(b);
  memcpy(&i, &ui, sizeof(uint32_t));
  return i;
}
void binarizer::code_int32(int32_t a0, char b[5]) {
  uint32_t ui;
  memcpy(&ui, &a0, sizeof(uint32_t));
  code_uint32(ui,b);
}

// ------------------- 64 bits -------------------
#ifdef HAVE_UINT64
uint64_t binarizer::decode_uint64(const char c[10]) {
  return
    uncod[(int)c[0]] * BASE9 +
    uncod[(int)c[1]] * BASE8 +
    uncod[(int)c[2]] * BASE7 +
    uncod[(int)c[3]] * BASE6 +
    uncod[(int)c[4]] * BASE5 +
    uncod[(int)c[5]] * BASE4 +
    uncod[(int)c[6]] * BASE3 +
    uncod[(int)c[7]] * BASE2 +
    uncod[(int)c[8]] * BASE1 +
    uncod[(int)c[9]];
}
void binarizer::code_uint64(uint64_t a0, char b[10]) {
  uint64_t a1,a2,a3,a4,a5,a6,a7,a8,a9;
  a1   = a0/BASE1;
  a2   = a0/BASE2;
  a3   = a0/BASE3;
  a4   = a0/BASE4;
  a5   = a0/BASE5;
  a6   = a0/BASE6;
  a7   = a0/BASE7;
  a8   = a0/BASE8;
  a9   = a0/BASE9;
  b[0] = cod[a9];
  b[1] = cod[a8-a9*BASE];
  b[2] = cod[a7-a8*BASE];
  b[3] = cod[a6-a7*BASE];
  b[4] = cod[a5-a6*BASE];
  b[5] = cod[a4-a5*BASE];
  b[6] = cod[a3-a4*BASE];
  b[7] = cod[a2-a3*BASE];
  b[8] = cod[a1-a2*BASE];
  b[9] = cod[a0-a1*BASE];
}

// signed case is reduced to unsigned case. Two complement
// representation is assumed
int64_t binarizer::decode_int64(const char b[10]) {
  int64_t i;
  uint64_t ui;
  ui = decode_uint64(b);
  memcpy(&i, &ui, sizeof(uint64_t));
  return i;
}
void binarizer::code_int64(int64_t a0, char b[10]) {
  uint64_t ui;
  memcpy(&ui, &a0, sizeof(uint64_t));
  code_uint64(ui,b);
}
#endif

// ------------------- FLOATING POING -------------------
void binarizer::code_float (float i, char b[5]) {
  // asumimos float en representación IEEE 754, lo pasamos bit a bit a
  // uint32
  uint32_t ui;
  memcpy(&ui, &i, sizeof(float));
  code_uint32(ui,b);
}


void binarizer::code_double(double i, char b[10]) {
  // asumimos double en representación IEEE 754, lo pasamos bit a bit
  // a uint64
  uint32_t words[2];
  uint32_t msw, lsw;
  memcpy(words, &i, sizeof(double));

  if (endianness == BIG_ENDIAN) {
    msw = words[0];
    lsw = words[1];
  } else {
    msw = words[1];
    lsw = words[0];
  }

  code_uint32(msw, b);
  code_uint32(lsw, b+5);
}

float binarizer::decode_float(const char b[5]) {
  // asumimos float en representación IEEE 754, lo pasamos bit a bit a
  // uint32
  float f;
  uint32_t ui;
  ui = decode_uint32(b);
  memcpy(&f, &ui, sizeof(uint32_t));
  return f;
}


double binarizer::decode_double(const char b[10]) {
  // asumimos double en representación IEEE 754, lo pasamos bit a bit
  // a uint64
  uint32_t msw, lsw;
  uint32_t words[2];
  double result;

  msw = decode_uint32(b);
  lsw = decode_uint32(b+5);
  if (endianness == BIG_ENDIAN) {
    words[0] = msw;
    words[1] = lsw;
  } else {
    words[0] = lsw;
    words[1] = msw;
  }
  memcpy(&result, words, sizeof(double));
  return result;
}


// ------------------- AUXILIAR -------------------

int binarizer::buffer_size_16(int num, bool with_newlines) {
  int estimation = num*3+1; // +1 due to final '\0'
  if (with_newlines) // \n every 26 numbers to get 78char linewith
    estimation += num/26+1;  
  return estimation;
}
int binarizer::buffer_size_32(int num, bool with_newlines) {
  int estimation = num*5+1; // +1 due to final '\0'
  if (with_newlines) // \n every 26 numbers to get 80char linewith
    estimation += num/16+1;  
  return estimation;
}
int binarizer::buffer_size_64(int num, bool with_newlines) {
  int estimation = num*10+1; // +1 due to final '\0'
  if (with_newlines) // \n every 26 numbers to get 80char linewith
    estimation += num/8+1;  
  return estimation;
}

// ------------------- AUXILIAR -------------------
int binarizer::code_vector_float(const float *vect,
				 int vect_size, 
				 char *dest_buffer,
				 int dest_buffer_size,
				 bool with_newlines) {
  return code_iterator_float(vect, vect+vect_size, vect_size,
			     dest_buffer, dest_buffer_size,
			     with_newlines);
}

int binarizer::code_vector_int32(const int32_t *vect,
				 int vect_size, 
				 char *dest_buffer,
				 int dest_buffer_size,
				 bool with_newlines) {
  int i;
  char *rb = dest_buffer;
  if (buffer_size_32(vect_size,with_newlines) <= dest_buffer_size) {
    i = 0;
    while (i<vect_size) {
      code_int32(vect[i],rb);
      rb+=5;
      i++;
      if (with_newlines && !(i%16)) { *rb = '\n'; rb++; }
    }
    if (with_newlines && (i % 16))  { *rb = '\n'; rb++; }
    *rb = '\0'; rb++;
  }
  return rb-dest_buffer;
}



