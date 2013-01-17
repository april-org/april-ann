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
#include "read_file_stream.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

bool ReadFileStream::moveAndFillBuffer() {
  // printf ("------------- MOVE ------------ %d %d\n", buffer_pos, buffer_len);
  int diff = buffer_len - buffer_pos;
  if (diff > 0)
    memcpy(buffer, buffer + buffer_pos, diff);
  if (!feof(f))
    buffer_len = fread(buffer + diff, sizeof(char), buffer_pos, f) + diff;
  else buffer_len = 0;
  buffer_pos = 0;
  return buffer_len != 0;
}
  
bool ReadFileStream::resizeAndFillBuffer() {
  // printf ("------------- RESIZE ------------\n");
  if (feof(f)) return false;
  unsigned int old_max_len = max_buffer_len;
  max_buffer_len *= 2;
  char *new_buffer = new char[max_buffer_len + 1];
  memcpy(new_buffer, buffer, old_max_len);
  buffer_len += fread(new_buffer + buffer_len, sizeof(char), max_buffer_len - buffer_len, f);
  delete[] buffer;
  buffer = new_buffer;
  return true;
}

bool ReadFileStream::trim(const char *delim) {
  while(strchr(delim, buffer[buffer_pos])) {
    ++buffer_pos;
    if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return false;
  }
  return true;
}

ReadFileStream::ReadFileStream(const char *path) {
  f		   = fopen(path, "r");
  buffer	   = new char[DEFAULT_BUFFER_LEN+1];
  max_buffer_len = DEFAULT_BUFFER_LEN;
  buffer_len	   = max_buffer_len;
  buffer_pos	   = max_buffer_len;
}

ReadFileStream::~ReadFileStream() {
  delete[] buffer;
  fclose(f);
}

constString ReadFileStream::getToken(const char *delim) {
  if (buffer_len == 0) return constString();
  // comprobamos que haya datos en el buffer
  if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return constString();
  // hacemos un trim de los delimitadores
  if (!trim(delim)) return constString();
  // last_pos apuntara al fina de la ejecucion al primer caracter
  // delimitador encontrado
  int last_pos = buffer_pos;
  do {
    ++last_pos;
    // si llegamos al final del buffer
    if (last_pos >= buffer_len) {
      // podemos hacer dos cosas, si se puede primero mover a la
      // izquierda y rellenar lo que queda, si no se puede entonces
      // aumentar el tamanyo del buffer y rellenar
      if (buffer_pos > 0) {
	last_pos -= buffer_pos;
	if (!moveAndFillBuffer()) break;
      }
      else if (!resizeAndFillBuffer()) break;
    }
    // printf ("buffer[%d] = %c -- buffer[%d] = %c\n", buffer_pos, buffer[buffer_pos],
    // last_pos, buffer[last_pos]);
  } while(strchr(delim, buffer[last_pos]) == 0);
  // en este momento last_pos apunta al primer caracter delimitador,
  // o al ultimo caracter del buffer
  // printf ("%d %d %d\n", buffer_pos, last_pos, buffer_len);
  if (strchr(delim, buffer[last_pos]) == 0) ++last_pos;
  buffer[last_pos] = '\0';
  const char *returned_buffer = buffer + buffer_pos;
  buffer_pos = last_pos + 1;
  return constString(returned_buffer);
}

constString ReadFileStream::extract_line() {
  return getToken("\n\r");
}

constString ReadFileStream::extract_u_line() {
  constString aux;
  do {
    aux = getToken("\r\n");
  } while ((aux.len() > 0) && (aux[0] == '#'));
  return aux;
}
