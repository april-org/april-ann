/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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

#include <cstdarg>
#include "file.h"

namespace april_io {
  
  bool File::moveAndFillBuffer() {
    // move to the left the remaining data
    int diff = buffer_len - buffer_pos;
    for (int i=0; i<diff; ++i) buffer[i] = buffer[buffer_pos + i];
    if (!stream->eof()) {
      // fill the right part with read data
      buffer_len = stream->read(buffer + diff, sizeof(char), buffer_pos) + diff;
    }
    else {
      buffer_len = 0;
    }
    buffer_pos = 0;
    // returns false if the buffer is empty, true otherwise
    return buffer_len != 0;
  }
  
  bool File::resizeAndFillBuffer() {
    // returns false if EOF
    if (stream->eof()) return false;
    unsigned int old_max_len = max_buffer_len;
    max_buffer_len <<= 1;
    char *new_buffer = new char[max_buffer_len + 1];
    // copies the data to the new buffer
    memcpy(new_buffer, buffer, old_max_len);
    // fills the right part with read data
    buffer_len += stream->read(new_buffer + buffer_len,
                               sizeof(char), (max_buffer_len - buffer_len));
    delete[] buffer;
    buffer = new_buffer;
    // returns true if not EOF
    return true;
  }
  
  bool File::trim(const char *delim) {
    while(strchr(delim, buffer[buffer_pos])) {
      ++buffer_pos;
      // returns false if EOF
      if (buffer_pos >= buffer_len && !moveAndFillBuffer()) return false;
    }
    // returns true if not EOF
    return true;
  }
  
  File::File(Stream *stream) : stream(stream) {
    total_bytes    = 0;
    buffer         = new char[april_io::DEFAULT_BUFFER_LEN];
    max_buffer_len = april_io::DEFAULT_BUFFER_LEN;
    setBufferAsFull();
    IncRef(stream);
  }
  
  File::~File() {
    delete[] buffer;
    DecRef(stream);
  }

  bool File::isOpened() const {
    return stream->isOpened();
  }
  
  bool File::good() const {
    return stream->isOpened() && (buffer_len>buffer_pos ||
                                  !stream->eof());
  }
    
  void File::close() {
    if (buffer != 0) {
      delete[] buffer;
      buffer = 0;
      buffer_len = 0;
      stream->close();
    }
  }

  constString File::getToken(int size) {
    if (buffer_len == 0) return constString();
    // comprobamos que haya datos en el buffer
    if (buffer_pos >= buffer_len && !moveAndFillBuffer()) {
      return constString();
    }
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
        else if (!resizeAndFillBuffer()) {
          break;
        }
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
  
  constString File::getToken(const char *delim) {
    if (buffer_len == 0) return constString();
    // comprobamos que haya datos en el buffer
    if (buffer_pos >= buffer_len && !moveAndFillBuffer()) {
      return constString();
    }
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
        else if (!resizeAndFillBuffer()) {
          break;
        }
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
  
  constString File::extract_line() {
    return getToken("\n\r");
  }

  constString File::extract_u_line() {
    constString aux;
    do {
      aux = getToken("\r\n");
    } while ((aux.len() > 0) && (aux[0] == '#'));
    return aux;
  }
  
  size_t File::write(const void *buffer, size_t len) {
    return stream->write(buffer, sizeof(char), len);
  }
  
  int File::printf(const char *format, ...) {
    va_list arg;
    char *aux_buffer;
    size_t len;
    va_start(arg, format);
    if (vasprintf(&aux_buffer, format, arg) < 0) {
      ERROR_EXIT(256, "Problem creating auxiliary buffer\n");
    }
    len = strlen(aux_buffer);
    if (len > 0) len = write(aux_buffer, len);
    free(aux_buffer);
    return len;
  }
  
  int File::seek(int whence, int offset) {
    if (buffer_len == 0) return 0;
    switch(whence) {
    case SEEK_SET:
      // In this case, a position in the stream is indicated, so it is easy to
      // throw away the buffer and move the stream cursor
      setBufferAsFull();
      return stream->seek(offset, whence);
      break;
    case SEEK_CUR:
      if (offset > buffer_len - buffer_pos) {
        offset -= buffer_len - buffer_pos;
        return stream->seek(offset, whence);
      }
      else {
        buffer_pos += offset;
        return stream->seek(0, SEEK_CUR) - buffer_len + buffer_pos;
      }
      break;
    case SEEK_END:
      setBufferAsFull();
      return stream->seek(offset, whence);
      break;
    default:
      ERROR_EXIT1(256, "Incorrect whence value %d to seek method\n", whence);
    }
    // default return, it will never happens
    return 0;
  }

  void File::flush() {
    stream->flush();
  }
    
  size_t File::getTotalBytes() const {
    return total_bytes;
  }

} // namespace april_io
