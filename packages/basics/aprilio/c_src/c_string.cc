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

namespace AprilIO {
  CStringStream::CStringStream() :
    StreamMemory(), in_pos(0), out_pos(0) {
    data.reserve(StreamMemory::BLOCK_SIZE);
  }

  CStringStream::CStringStream(const AprilUtils::string &str) :
    StreamMemory(), data(str), in_pos(0), out_pos(data.size()) {
  }

  CStringStream::CStringStream(const char *str, size_t size) :
    StreamMemory(), data(str, size), in_pos(0), out_pos(size) {
  }
  
  CStringStream::~CStringStream() {
    close();
  }

  AprilUtils::constString CStringStream::getConstString() const {
    return AprilUtils::constString(data.c_str(), size());
  }

  void CStringStream::swapString(AprilUtils::string &other) {
    data.swap(other);
  }
  
  char *CStringStream::releaseString() {
    resetBuffers();
    in_pos = out_pos = 0;
    return data.release();
  }
  
  bool CStringStream::empty() const {
    return data.empty();
  }
  
  size_t CStringStream::size() const {
    return data.size();
  }

  size_t CStringStream::capacity() const {
    return data.capacity()-1;
  }
  
  char CStringStream::operator[](size_t pos) const {
    april_assert(pos < size());
    return data[pos];
  }

  char &CStringStream::operator[](size_t pos) {
    april_assert(pos < size());
    return data[pos];
  }
  
  void CStringStream::clear() {
    resetBuffers();
    data.clear();
    in_pos = out_pos = 0;
  }
  
  int CStringStream::push(lua_State *L) {
    lua_pushlstring(L, data.c_str(), size());
    return 1;
  }
  
  bool CStringStream::isOpened() const {
    return true;
  }
  
  void CStringStream::close() {
    flush();
    // FIXME: closed = true; ???
  }
  
  off_t CStringStream::seek(int whence, long offset) {
    if (whence == SEEK_CUR && offset == 0) return size();
    off_t aux_pos = 0;
    switch(whence) {
    case SEEK_SET:
      aux_pos = offset;
      break;
    case SEEK_CUR:
      aux_pos = out_pos + getOutBufferPos() + offset;
      break;
    case SEEK_END:
      aux_pos = size() + offset;
      break;
    }
    ERROR_EXIT(128, "NOT IMPLEMENTED BEHAVIOR\n");
    if (aux_pos < 0) aux_pos = 0;
    else if (static_cast<size_t>(aux_pos) > size()) {
      aux_pos = static_cast<off_t>(size());
    }
    in_pos = out_pos = static_cast<size_t>(aux_pos);
    return aux_pos;
  }
  
  void CStringStream::flush() {
    size_t new_size = out_pos + getOutBufferPos();
    // april_assert(capacity() >= new_size);
    if (size() < new_size+1) {
      data.reserve(new_size+1);
      data.resize(new_size);
    }
    data[size()] = '\0';
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
    return StreamInterface::NO_ERROR_STRING;
  }
  
  const char *CStringStream::nextInBuffer(size_t &buf_len) {
    april_assert(size() >= in_pos + getInBufferPos());
    in_pos += getInBufferPos();
    buf_len = size() - in_pos;
    return data.c_str() + in_pos;
  }
  
  char *CStringStream::nextOutBuffer(size_t &buf_len) {
    out_pos += getOutBufferPos();
    if (out_pos >= capacity()) {
      // NOTE: Here we use data.capacity() which is capacity()+1.
      data.reserve(data.capacity() << 1);
      in_pos += getInBufferPos();
      resetBuffers();
    }
    buf_len = capacity() - out_pos;
    return data.begin() + out_pos;
  }
  
  bool CStringStream::eofStream() const {
    return in_pos >= size();
  }
  
  void CStringStream::moveOutBuffer(size_t len) {
    StreamBuffer::moveOutBuffer(len);
    flush();
  }
} // namespace AprilIO
