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
#include "april_assert.h"
#include "c_string.h"

namespace april_io {
  CStringStream::CStringStream() :
    StreamMemory(), in_pos(0), out_pos(0) {
    data.reserve(StreamMemory::BLOCK_SIZE);
  }

  CStringStream::CStringStream(const april_utils::string &str) :
    StreamMemory(), data(str), in_pos(0), out_pos(data.size()) {
  }

  CStringStream::CStringStream(const char *str, size_t size) :
    StreamMemory(), data(str, size), in_pos(0), out_pos(size) {
  }
  
  CStringStream::~CStringStream() {
    close();
  }

  constString CStringStream::getConstString() const {
    return constString(data.c_str(), data.size());
  }
  
  bool CStringStream::empty() const {
    return data.empty();
  }
  
  size_t CStringStream::size() const {
    return data.size();
  }
  
  char CStringStream::operator[](size_t pos) const {
    april_assert(pos < data.size());
    return data[pos];
  }

  char &CStringStream::operator[](size_t pos) {
    april_assert(pos < data.size());
    return data[pos];
  }
  
  void CStringStream::clear() {
    resetBuffers();
    data.clear();
    in_pos = out_pos = 0;
  }
  
  int CStringStream::push(lua_State *L) {
    lua_pushlstring(L, data.c_str(), data.size());
    return 1;
  }
  
  bool CStringStream::isOpened() const {
    return true;
  }
  
  void CStringStream::close() {
    clear();
  }
  
  off_t CStringStream::seek(int whence, int offset) {
    UNUSED_VARIABLE(whence);
    UNUSED_VARIABLE(offset);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return 0;
  }
  
  void CStringStream::flush() {
  }
  
  int CStringStream::setvbuf(int mode, size_t size) {
    UNUSED_VARIABLE(mode);
    UNUSED_VARIABLE(size);
    return 0;
  }
  
  bool CStringStream::hasError() const {
    return false;
  }
  
  const char *CStringStream::getErrorMsg() const {
    return StreamMemory::NO_ERROR_STRING;
  }
  
  const char *CStringStream::nextInBuffer(size_t &buf_len) {
    buf_len = data.size() - in_pos - getInBufferPos();
    return data.c_str() + in_pos + getInBufferPos();
  }
  
  char *CStringStream::nextOutBuffer(size_t &buf_len) {
    if (out_pos + getOutBufferPos() >= data.capacity()) {
      data.reserve(data.capacity() << 1);
    }
    in_pos  += getInBufferPos();
    out_pos += getOutBufferPos();
    resetBuffers();
    buf_len = data.capacity() - out_pos;
    return data.begin() + out_pos;
  }
  
  bool CStringStream::eofStream() const {
    return in_pos + getInBufferPos() >= data.size();
  }
} // namespace april_io
