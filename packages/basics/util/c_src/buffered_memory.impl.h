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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "error_print.h"
#include "buffered_memory.h"

#define DEFAULT_BUFFER_LEN 4096

template<typename MEMORY_TYPE>
bool BufferedMemory<MEMORY_TYPE>::moveAndFillBuffer() {
  // move to the left the remaining data
  int diff = buffer_len - buffer_pos;
  for (int i=0; i<diff; ++i) buffer[i] = buffer[buffer_pos + i];
  if (!memory.eofS())
    // fill the right part with read data
    buffer_len = memory.readS(buffer + diff, sizeof(char), buffer_pos) + diff;
  else buffer_len = 0;
  buffer_pos = 0;
  // returns false if the buffer is empty, true otherwise
  return buffer_len != 0;
}

template<typename MEMORY_TYPE>
bool BufferedMemory<MEMORY_TYPE>::resizeAndFillBuffer() {
  // returns false if EOF
  if (memory.eofS()) return false;
  unsigned int old_max_len = max_buffer_len;
  max_buffer_len <<= 1;
  char *new_buffer = new char[max_buffer_len + 1];
  // copies the data to the new buffer
  memcpy(new_buffer, buffer, old_max_len);
  // fills the right part with read data
  buffer_len += memory.readS(new_buffer + buffer_len,
			     sizeof(char), (max_buffer_len - buffer_len));
  delete[] buffer;
  buffer = new_buffer;
  // returns true if not EOF
  return true;
}

template<typename MEMORY_TYPE>
bool BufferedMemory<MEMORY_TYPE>::trim(const char *delim) {
  while(strchr(delim, buffer[buffer_pos])) {
    ++buffer_pos;
    // returns false if EOF
    if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return false;
  }
  // returns true if not EOF
  return true;
}

template<typename MEMORY_TYPE>
BufferedMemory<MEMORY_TYPE>::BufferedMemory(const char *path, const char *mode) :
  Referenced(), memory() {
  total_bytes    = 0;
  buffer         = new char[DEFAULT_BUFFER_LEN];
  max_buffer_len = DEFAULT_BUFFER_LEN;
  setBufferAsFull();
  if (!memory.openS(path, mode))
    ERROR_EXIT2(256, "Unable to open path %s with mode %s\n",
		path, mode);
}

template<typename MEMORY_TYPE>
BufferedMemory<MEMORY_TYPE>::~BufferedMemory() {
  close();
}

template<typename MEMORY_TYPE>
void BufferedMemory<MEMORY_TYPE>::close() {
  if (buffer != 0) {
    delete[] buffer;
    buffer = 0;
    memory.closeS();
  }
}

template<typename MEMORY_TYPE>
void BufferedMemory<MEMORY_TYPE>::flush() {
  memory.flushS();
}

template<typename MEMORY_TYPE>
int BufferedMemory<MEMORY_TYPE>::seek(int whence, int offset) {
  switch(whence) {
  case SEEK_SET:
    // In this case, a position in the memory is indicated, so it is easy to
    // throw away the buffer and move the memory cursor
    setBufferAsFull();
    return memory.seekS(offset, whence);
    break;
  case SEEK_CUR:
    if (offset > buffer_len - buffer_pos) {
      offset -= buffer_len - buffer_pos;
      return memory.seekS(offset, whence);
    }
    else {
      buffer_pos += offset;
      return memory.seekS(0, SEEK_CUR) - buffer_len + buffer_pos;
    }
    break;
  case SEEK_END:
    setBufferAsFull();
    return memory.seekS(offset, whence);
    break;
  default:
    ERROR_EXIT1(256, "Incorrect whence value %d to seek method\n", whence);
  }
  return 0;
}

template<typename MEMORY_TYPE>
int BufferedMemory<MEMORY_TYPE>::readAndPushNumberToLua(lua_State *L) {
  constString token = getToken(" ,;\t\n\r");
  if (token.empty()) return 0;
  double number;
  if (!token.extract_double(&number))
    ERROR_EXIT(256, "Impossible to extract a number from current file pos\n");
  lua_pushnumber(L, number);
  return 1;
}

template<typename MEMORY_TYPE>
int BufferedMemory<MEMORY_TYPE>::readAndPushStringToLua(lua_State *L, int size) {
  constString token = getToken(size);
  if (token.empty()) return 0;
  lua_pushlstring(L, (const char *)(token), token.len());
  return 1;
}

/*
  int BufferedMemory<MEMORY_TYPE>::readAndPushCharToLua(lua_State *L) {
  char ch = getChar();
  lua_pushlstring(L, &ch, 1);
  return 1;
  }
*/

template<typename MEMORY_TYPE>
int BufferedMemory<MEMORY_TYPE>::readAndPushLineToLua(lua_State *L) {
  constString line = extract_line();
  if (line.empty()) return 0;
  lua_pushlstring(L, (const char *)(line), line.len());
  return 1;
}

template<typename MEMORY_TYPE>
int BufferedMemory<MEMORY_TYPE>::readAndPushAllToLua(lua_State *L) {
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
  char BufferedMemory<MEMORY_TYPE>::getChar() {
  if (buffer_len == 0) return EOF;
  // comprobamos que haya datos en el buffer
  if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return EOF;
  char ch = buffer[buffer_pos];
  ++buffer_pos;
  ++total_bytes;
  return ch;
  }
*/

template<typename MEMORY_TYPE>
constString BufferedMemory<MEMORY_TYPE>::getToken(int size) {
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

template<typename MEMORY_TYPE>
constString BufferedMemory<MEMORY_TYPE>::getToken(const char *delim) {
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

template<typename MEMORY_TYPE>
constString BufferedMemory<MEMORY_TYPE>::extract_line() {
  return getToken("\n\r");
}

template<typename MEMORY_TYPE>
constString BufferedMemory<MEMORY_TYPE>::extract_u_line() {
  constString aux;
  do {
    aux = getToken("\r\n");
  } while ((aux.len() > 0) && (aux[0] == '#'));
  return aux;
}

template<typename MEMORY_TYPE>
void BufferedMemory<MEMORY_TYPE>::printf(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  total_bytes += memory.printfS(format, ap);
  va_end(ap);
}

template<typename MEMORY_TYPE>
void BufferedMemory<MEMORY_TYPE>::write(const void *buffer, size_t len) {
  total_bytes += memory.writeS(buffer, sizeof(char), len);
}
