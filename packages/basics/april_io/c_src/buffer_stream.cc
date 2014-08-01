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
#include <cstring>

#include "error_print.h"
#include "buffer_stream.h"
#include "unused_variable.h"

namespace april_io {
        
  OutputBufferStream::OutputBufferStream() : buffer(0), pos(0), expected_size(0) {
  }
  
  OutputBufferStream::~OutputBufferStream() {
    close();
    delete[] buffer;
  }

  void OutputBufferStream::close() {
  }
  
  void OutputBufferStream::flush() { }
        
  bool OutputBufferStream::isOpened() const { return buffer != 0; }
  
  bool OutputBufferStream::eof() {
    return static_cast<size_t>(pos - buffer) == expected_size;
  }
        
  int OutputBufferStream::seek(long offset, int whence) {
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(whence);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
        
  size_t OutputBufferStream::read(void *ptr, size_t size, size_t nmemb) {
    UNUSED_VARIABLE(ptr);
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(nmemb);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
        
  size_t OutputBufferStream::write(const void *ptr, size_t size, size_t nmemb) {
    size_t len = nmemb*size;
    if (len + static_cast<size_t>(pos - buffer) > expected_size) {
      ERROR_EXIT(128, "OutputBuffer overflow\n");
    }
    memcpy(pos, ptr, len);
    pos += len;
    return len;
  }

  void OutputBufferStream::setExpectedSize(size_t sz) {
    expected_size = sz;
    buffer = new char[sz];
    pos = buffer;
    if (buffer == 0) ERROR_EXIT(256, "Impossible to allocate the buffer\n");
  }

  char *OutputBufferStream::getBufferProperty() {
    char *aux = buffer;
    buffer = pos = 0;
    return aux;
  }

  constString OutputBufferStream::get() const {
    return constString(buffer, pos-buffer);
  }

  //////////////////////////////////////////////////////////////////////////

  InputBufferStream::InputBufferStream(const char *buffer, size_t n) :
    buffer(buffer, n) {
  }
  
  InputBufferStream::~InputBufferStream() {
    close();
  }

  void InputBufferStream::close() {
  }
  
  void InputBufferStream::flush() { }
        
  bool InputBufferStream::isOpened() const { return true; }
  
  bool InputBufferStream::eof() {
    return buffer.len() > 0;
  }
        
  int InputBufferStream::seek(long offset, int whence) {
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(whence);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }
        
  size_t InputBufferStream::read(void *ptr, size_t size, size_t nmemb) {
    constString result = buffer.extract_prefix(size*nmemb);
    memcpy(ptr, (const char *)result, result.len());
    return result.len();
  }
        
  size_t InputBufferStream::write(const void *ptr, size_t size, size_t nmemb) {
    UNUSED_VARIABLE(ptr);
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(nmemb);
    ERROR_EXIT(128, "Not implemented\n");
    return 0;
  }

} // namespace april_io
