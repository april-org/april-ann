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
#include <cstring>
#include <cstdlib>
#include "constString.h"
#include "binarizer.h"

constString::constString(const char *s, size_t n) {
  buffer = s;
  length = n;
}

constString::constString(const char *s) {
  buffer = s;
  length = strlen(s);
}

char* constString::newString() const {
  int  addzero = (length == 0 || buffer[length-1] != '\0') ? 1 : 0;
  char *resul  = new char[length + addzero];
  strncpy(resul, buffer, length);
  if (addzero) resul[length] = '\0';
  return resul;
}

bool constString::operator == (const constString &otro) const {
  if (length != otro.length) return false;
  return strncmp(buffer,otro.buffer,length) == 0;
}

bool constString::operator != (const constString &otro) const {
  if (length != otro.length) return true;
  return strncmp(buffer,otro.buffer,length) != 0;
}

bool constString::operator <  (const constString &otro) const {
  size_t min = (length < otro.length) ? length : otro.length;
  int cmp = strncmp(buffer,otro.buffer,min);
  // Si difieren en los primeros min caracteres, ya esta
  if (cmp==-1) return true;
  else if (cmp==1)  return false;
  // si no, miramos la longitud
  else return (length < otro.length);
}

bool constString::operator <= (const constString &otro) const {
  size_t min = (length < otro.length) ? length : otro.length;
  int cmp = strncmp(buffer,otro.buffer,min);
  // Si difieren en los primeros min caracteres, ya esta
  if (cmp==-1) return true;
  else if (cmp==1)  return false;
  // si no, miramos la longitud
  else return(length <= otro.length);
}

bool constString::operator >  (const constString &otro) const {
  size_t min = (length < otro.length) ? length : otro.length;
  int cmp = strncmp(buffer,otro.buffer,min);
  // Si difieren en los primeros min caracteres, ya esta
  if (cmp==-1) return false;
  else if (cmp==1)  return true;
  // si no, miramos la longitud
  else return (length > otro.length);
}

bool constString::operator >=  (const constString &otro) const {
  size_t min = (length < otro.length) ? length : otro.length;
  int cmp = strncmp(buffer,otro.buffer,min);
  // Si difieren en los primeros min caracteres, ya esta
  if (cmp==-1) return false;
  else if (cmp==1)  return true;
  // si no, miramos la longitud
  else return (length >= otro.length);
}

void constString::skip(size_t n) {
  if (buffer) {
    if (n>length) n = length;
    length -= n;
    buffer += n;
  }
}

void constString::ltrim(const char *acepta) {
  // todo: puede fallar si la cadena no es terminada en '\0'
  if (buffer) {
    size_t n = strspn(buffer,acepta);
    if (n>length) n=length;
    length -= n;
    buffer += n;
  }
}

bool constString::is_prefix(const constString &theprefix) {
  if (length < theprefix.length) 
    return false;
  return strncmp(buffer,theprefix.buffer,theprefix.length) == 0;
}

bool constString::is_prefix(const char *theprefix) {
  return is_prefix(constString(theprefix));
}

bool constString::skip(const constString &theprefix) {
  if (!is_prefix(theprefix)) return false;
  skip(theprefix.length);
  return true;
}

bool constString::skip(const char *theprefix) {
  return skip(constString(theprefix));
}

constString constString::extract_prefix(size_t prefix_len) {
  if (empty()) return constString();
  // prefix_len no puede ser mayor que length:
  if (prefix_len > length) prefix_len = length;
  constString resul = constString(buffer,prefix_len);
  length -= prefix_len;
  buffer += prefix_len;
  return resul;
}

constString constString::extract_token(const char *separadores) {
  ltrim(separadores);
  if (empty()) return constString();
  const char *newbuffer = buffer;
  size_t newlength = strcspn(newbuffer,separadores);
  if (newlength>length) newlength=length;
  skip(newlength);
  return constString(newbuffer,newlength);
}

constString constString::extract_line() {
  return extract_token("\r\n");
}

constString constString::extract_u_line() {
  constString aux;
  do {
    aux = extract_token("\r\n");
  } while ((aux.length > 0) && (aux.buffer[0] == '#'));
  return aux;
}

bool constString::extract_char(char *resul) {
  if (empty()) return false;
  *resul = buffer[0];
  skip(1);
  return true;
}

bool constString::extract_float(float *resul,const char *separadores) {
  ltrim(separadores);
  if (empty()) return false;
  char *aux;
  // fixme: strtof no controla si nos salimos del constString, luego
  // lo comprobamos viendo si length < 0 pero podría dar violación de
  // segmento
  float value = strtof(buffer,&aux);  
  if (aux == buffer) return false;
  if (aux > buffer+length) return false;
  length -= aux-buffer;
  buffer = aux;
  *resul = value;
  return true;
}

bool constString::extract_float_binary(float *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<5 || buffer[0] == 0 || buffer[1] == 0 ||
      buffer[2] == 0 || buffer[3] == 0 || buffer[4] == 0) 
    return false;
  *resul = binarizer::decode_float(buffer);
  skip(5);
  return true;  
}

bool constString::extract_double(double *resul,const char *separadores) {
  ltrim(separadores);
  char *aux;
  // fixme: strtod no controla si nos salimos del constString, luego
  // lo comprobamos viendo si length < 0 pero podría dar violación de
  // segmento
  double value = strtod(buffer,&aux);
  if (aux == buffer) return false;
  if (aux > buffer+length) return false;
  length -= aux-buffer;
  buffer = aux;
  *resul = value;
  return true;
}

bool constString::extract_log_float_binary(log_float *resul) {
  float aux;
  if (extract_float_binary(&aux)) {
    *resul = log_float(aux);
    return true;
  }
  return false;
}

bool constString::extract_log_double_binary(log_double *resul) {
  double aux;
  if (extract_double_binary(&aux)) {
    *resul = log_double::from_double(aux);
    return true;
  }
  return false;
}

bool constString::extract_double_binary(double *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<5 || buffer[0] == 0 ||
      buffer[1] == 0 || buffer[2] == 0 || buffer[3] == 0 || 
      buffer[4] == 0 || buffer[5] == 0 || buffer[6] == 0 ||
      buffer[7] == 0 || buffer[8] == 0 || buffer[9] == 0) 
    return false;
  *resul = binarizer::decode_double(buffer);
  skip(10);
  return true;  
}

bool constString::extract_int(int *resul, int base,const char *separadores) {
  long int x;
  if (extract_long(&x,base,separadores)) { *resul = (int)x; return true; }
  return false;
}

bool constString::extract_unsigned_int(unsigned int *resul, int base,const char *separadores) {
  long int x;
  if (extract_long(&x,base,separadores)) { *resul = (unsigned int)x; return true; }
  return false;
}

bool constString::extract_long(long int *resul, int base,
			       const char *separadores) {
  ltrim(separadores);
  if (empty()) return false;
  char *aux;
  // fixme: strtol no controla si nos salimos del constString, luego
  // lo comprobamos viendo si length < 0 pero podría dar violación de
  // segmento
  long int value = strtol(buffer,&aux, base);
  if (aux == buffer) return false;
  if (aux > buffer+length) return false;
  length -= aux-buffer;
  buffer = aux;
  *resul = value;
  return true;
}

bool constString::extract_long_long(long long int *resul, int base,
				    const char *separadores) {
  ltrim(separadores);
  if (empty()) return false;
  char *aux;
  long long int value = strtoll(buffer,&aux, base);
  if (aux == buffer) return false;
  if (aux > buffer+length) return false;
  length -= aux-buffer;
  buffer = aux;
  *resul = value;
  return true;
}

#ifdef HAVE_UINT16
bool constString::extract_uint16_binary(uint16_t *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<3 || buffer[0] == 0 || buffer[1] == 0 || buffer[2] == 0)
    return false;
  *resul = binarizer::decode_uint16(buffer);
  skip(3);
  return true;  
}

bool constString::extract_int16_binary(int16_t *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<3 || buffer[0] == 0 || buffer[1] == 0 || buffer[2] == 0)
    return false;
  *resul = binarizer::decode_int16(buffer);
  skip(3);
  return true;  
}
#endif

bool constString::extract_uint32_binary(uint32_t *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<5 || buffer[0] == 0 || buffer[1] == 0 ||
      buffer[2] == 0 || buffer[3] == 0 || buffer[4] == 0) 
    return false;
  *resul = binarizer::decode_uint32(buffer);
  skip(5);
  return true;  
}

bool constString::extract_int32_binary(int32_t *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<5 || buffer[0] == 0 || buffer[1] == 0 ||
      buffer[2] == 0 || buffer[3] == 0 || buffer[4] == 0)
    return false;
  *resul = binarizer::decode_int32(buffer);
  skip(5);
  return true;  
}

#ifdef HAVE_UINT64
bool constString::extract_uint64_binary(uint64_t *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<10 || buffer[0] == 0 || buffer[1] == 0 ||
      buffer[2] == 0 || buffer[3] == 0 || buffer[4] == 0 ||
      buffer[5] == 0 || buffer[6] == 0 || buffer[7] == 0 ||
      buffer[8] == 0 || buffer[9] == 0)
    return false;
  *resul = binarizer::decode_uint64(buffer);
  skip(10);
  return true;  
}

bool constString::extract_int64_binary(int64_t *resul) {
  if (empty()) return false;
  if (buffer[0] == '\n') skip(1);
  if (length<10 || buffer[0] == 0 || buffer[1] == 0 ||
      buffer[2] == 0 || buffer[3] == 0 || buffer[4] == 0 ||
      buffer[5] == 0 || buffer[6] == 0 || buffer[7] == 0 ||
      buffer[8] == 0 || buffer[9] == 0)
    return false;
  *resul = binarizer::decode_int64(buffer);
  skip(10);
  return true;  
}
#endif

void constString::print(FILE *F) {
  for (size_t i=0;i<length;i++) 
    fputc(buffer[i],F);
}

char *copystr(const char *c) {
  if (c == 0) return 0;
  char *n = new char[strlen(c)+1];
  strcpy(n,c);
  return n;
}

