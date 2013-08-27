/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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

// This wrapper is inspired by GZIO http://luaforge.net/projects/gzio/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "error_print.h"
#include "gzfile_wrapper.h"

#define DEFAULT_BUFFER_LEN 4096

bool GZFileWrapper::moveAndFillBuffer() {
  // printf ("------------- MOVE ------------ %d %d\n", buffer_pos, buffer_len);
  int diff = buffer_len - buffer_pos;
  for (int i=0; i<diff; ++i) buffer[i] = buffer[buffer_pos + i];
  if (!gzeof(f))
    buffer_len = gzread(f, buffer + diff, sizeof(char)*buffer_pos) + diff;
  else buffer_len = 0;
  buffer_pos = 0;
  return buffer_len != 0;
}
  
bool GZFileWrapper::resizeAndFillBuffer() {
  // printf ("------------- RESIZE ------------\n");
  if (gzeof(f)) return false;
  unsigned int old_max_len = max_buffer_len;
  max_buffer_len *= 2;
  char *new_buffer = new char[max_buffer_len + 1];
  memcpy(new_buffer, buffer, old_max_len);
  buffer_len += gzread(f, new_buffer + buffer_len,
		       sizeof(char)*(max_buffer_len - buffer_len));
  delete[] buffer;
  buffer = new_buffer;
  return true;
}

bool GZFileWrapper::trim(const char *delim) {
  while(strchr(delim, buffer[buffer_pos])) {
    ++buffer_pos;
    if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return false;
  }
  return true;
}

/*
  Opens a gzip (.gz) file for reading or writing. The mode parameter is as in
  fopen ("rb" or "wb") but can also include a compression level ("wb9") or a
  strategy: 'f' for filtered data as in "wb6f", 'h' for Huffman-only compression
  as in "wb1h", 'R' for run-length encoding as in "wb1R", or 'F' for fixed code
  compression as in "wb9F". (See the description of deflateInit2 for more
  information about the strategy parameter.) 'T' will request transparent
  writing or appending with no compression and not using the gzip format.

  'a' can be used instead of 'w' to request that the gzip stream that will be
  written be appended to the file. '+' will result in an error, since reading
  and writing to the same gzip file is not supported. The addition of 'x' when
  writing will create the file exclusively, which fails if the file already
  exists. On systems that support it, the addition of 'e' when reading or
  writing will set the flag to close the file on an execve() call.

  These functions, as well as gzip, will read and decode a sequence of gzip
  streams in a file. The append function of gzopen() can be used to create such
  a file. (Also see gzflush() for another way to do this.) When appending,
  gzopen does not test whether the file begins with a gzip stream, nor does it
  look for the end of the gzip streams to begin appending. gzopen will simply
  append a gzip stream to the existing file.

  gzopen can be used to read a file which is not in gzip format; in this case
  gzread will directly read from the file without decompression. When reading,
  this will be detected automatically by looking for the magic two-byte gzip
  header.

  gzopen returns NULL if the file could not be opened, if there was insufficient
  memory to allocate the gzFile state, or if an invalid mode was specified (an
  'r', 'w', or 'a' was not provided, or '+' was provided). errno can be checked
  to determine if the reason gzopen failed was that the file could not be
  opened.
*/
GZFileWrapper::GZFileWrapper(const char *path, const char *mode) : Referenced(){
  total_bytes    = 0;
  buffer         = new char[DEFAULT_BUFFER_LEN];
  max_buffer_len = DEFAULT_BUFFER_LEN;
  setBufferAsFull();
  f              = gzopen(path, mode);
  if (f == 0) ERROR_EXIT2(256, "Unable to open path %s with mode %s\n",
			  path, mode);
}

GZFileWrapper::~GZFileWrapper() {
  close();
}

void GZFileWrapper::close() {
  if (buffer != 0) {
    delete[] buffer;
    buffer = 0;
    gzclose(f);
  }
}

void GZFileWrapper::flush() {
  gzflush(f, Z_SYNC_FLUSH);
}

int GZFileWrapper::seek(int whence, int offset) {
  switch(whence) {
  case SEEK_SET:
    setBufferAsFull();
    return gzseek(f, offset, whence);
    break;
  case SEEK_CUR:
    if (offset > buffer_len - buffer_pos) {
      offset -= buffer_len - buffer_pos;
      return gzseek(f, offset, whence);
    }
    else {
      buffer_pos += offset;
      return gzseek(f, 0, SEEK_CUR) - buffer_len + buffer_pos;
    }
    break;
  default:
    ERROR_EXIT1(256, "Incorrect whence value %d to seek method\n", whence);
  }
}

int GZFileWrapper::readAndPushNumberToLua(lua_State *L) {
  constString token = getToken(" ,;\t\n\r");
  if (token.empty()) return 0;
  double number;
  if (!token.extract_double(&number))
    ERROR_EXIT(256, "Impossible to extract a number from current file pos\n");
  lua_pushnumber(L, number);
  return 1;
}


int GZFileWrapper::readAndPushStringToLua(lua_State *L, int size) {
  constString token = getToken(size);
  if (token.empty()) return 0;
  lua_pushlstring(L, (const char *)(token), token.len());
  return 1;
}

/*
  int GZFileWrapper::readAndPushCharToLua(lua_State *L) {
  char ch = getChar();
  lua_pushlstring(L, &ch, 1);
  return 1;
  }
*/

int GZFileWrapper::readAndPushLineToLua(lua_State *L) {
  constString line = extract_line();
  if (line.empty()) return 0;
  lua_pushlstring(L, (const char *)(line), line.len());
  return 1;
}

int GZFileWrapper::readAndPushAllToLua(lua_State *L) {
  constString line = getToken(1024);
  if (line.empty()) return 0;
  luaL_Buffer lua_buffer;
  luaL_buffinit(L, &lua_buffer);
  luaL_addlstring(&lua_buffer, (const char*)(line), line.len());
  while((line = getToken(1024)) && !line.empty())
    luaL_addlstring(&lua_buffer, (const char*)(line), line.len());
  luaL_pushresult(&lua_buffer);
  return 1;
}

/*
  char GZFileWrapper::getChar() {
  if (buffer_len == 0) return EOF;
  // comprobamos que haya datos en el buffer
  if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return EOF;
  char ch = buffer[buffer_pos];
  ++buffer_pos;
  ++total_bytes;
  return ch;
  }
*/

constString GZFileWrapper::getToken(int size) {
  if (buffer_len == 0) return constString();
  // comprobamos que haya datos en el buffer
  if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return constString();
  // last_pos apuntara al fina de la ejecucion al primer caracter
  // delimitador encontrado
  int last_pos = buffer_pos;
  while(last_pos - buffer_pos + 1 < size) {
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
  }
  // en este momento last_pos apunta al primer caracter delimitador,
  // o al ultimo caracter del buffer
  // printf ("%d %d %d\n", buffer_pos, last_pos, buffer_len);
  const char *returned_buffer = buffer + buffer_pos;
  size_t len   = last_pos - buffer_pos + 1;
  total_bytes += len;
  buffer_pos   = last_pos + 1;
  ++last_pos;
  return constString(returned_buffer, len);
}

constString GZFileWrapper::getToken(const char *delim) {
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
  total_bytes += buffer_pos - last_pos + 1;
  buffer_pos = last_pos + 1;
  return constString(returned_buffer);
}

constString GZFileWrapper::extract_line() {
  return getToken("\n\r");
}

constString GZFileWrapper::extract_u_line() {
  constString aux;
  do {
    aux = getToken("\r\n");
  } while ((aux.len() > 0) && (aux[0] == '#'));
  return aux;
}

void GZFileWrapper::printf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  char *aux_buffer;
  size_t len;
  if (vasprintf(&aux_buffer, format, ap) < 0)
    ERROR_EXIT(256, "Problem creating auxiliary buffer\n");
  len = strlen(aux_buffer);
  if (len > 0) total_bytes += gzwrite(f, aux_buffer, len);
  free(aux_buffer);
  va_end(ap);
}
