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
#ifndef BUFFERED_STREAM_H
#define BUFFERED_STREAM_H

#include "referenced.h"
#include "constString.h"
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

/**
   The STREAM_TYPE must complain with this interface
   
   STREAM_TYPE() default constructor
   bool   openS(const char *path, const char *mode);
   void   closeS();
   size_t readS(void *ptr, size_t size, size_t nmemb) = 0;
   size_t writeS(const void *ptr, size_t size, size_t nmemb) = 0;
   int    seekS(long offset, int whence) = 0;
   void   flushS() = 0;
   int    printfS(const char *format, va_list &arg) = 0;
   bool   eofS() = 0;
**/

/// This class is useful to define classes for input/output routines, as for
/// FILE or gzFile formats. The STREAM_TYPE must be a class or struct which
/// complains the interface defined in the file buffered_stream.h
template<typename STREAM_TYPE>
class BufferedStream : public Referenced {
  int total_bytes;
  char *buffer;
  int max_buffer_len, buffer_pos, buffer_len;
  STREAM_TYPE stream;
  
  bool moveAndFillBuffer();
  bool resizeAndFillBuffer();
  bool trim(const char *delim);
  
  void setBufferAsFull() {
    // forces to read from the file in the next getToken
    buffer_pos = max_buffer_len;
    buffer_len = max_buffer_len;
  }

public:

  BufferedStream(const char *path, const char *mode);
  ~BufferedStream();
  
  void close();
  void flush();
  int seek(int whence, int offset);
  
  /** Lua methods **/
  int readAndPushNumberToLua(lua_State *L);
  //  int readAndPushCharToLua(lua_State *L);
  int readAndPushStringToLua(lua_State *L, int size);
  int readAndPushLineToLua(lua_State *L);
  int readAndPushAllToLua(lua_State *L);
  /*****************/
  
  /** For matrix Read/Write C++ interface **/

  // char getChar();
  // be careful, this method returns a pointer to internal buffer
  constString getToken(int size);
  // be careful, this method returns a pointer to internal buffer
  constString getToken(const char *delim);
  // be careful, this method returns a pointer to internal buffer
  constString extract_line();
  // be careful, this method returns a pointer to internal buffer
  constString extract_u_line();
  
  void printf(const char *format, ...);

  void setExpectedSize(int sz) const { }
  
  int getTotalBytes() const { return total_bytes; }
  /**************************************/
};

// Includes the implementation
#include "buffered_stream.impl.h"

#endif // BUFFERED_STREAM_H
