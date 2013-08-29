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
#ifndef BUFFERED_MEMORY_H
#define BUFFERED_MEMORY_H

#include "referenced.h"
#include "constString.h"
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

/**
   The MEMORY_TYPE must complain with this interface
   
   MEMORY_TYPE() default constructor
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
/// FILE or gzFile formats. The MEMORY_TYPE must be a class or struct which
/// complains the interface defined in the file buffered_memory.h
template<typename MEMORY_TYPE>
class BufferedMemory : public Referenced {
  /// It measures the number of bytes written or read
  int total_bytes;
  /// Reading buffer, writings are done directly
  char *buffer;
  /// Reserved size of the buffer
  int max_buffer_len;
  /// Current position of first valid char
  int buffer_pos;
  /// Number of chars chars read
  int buffer_len;
  /// The memory which has implemented methods: openS, closeS, readS, writeS, seekS, flushS, printfS, eofS
  MEMORY_TYPE memory;
  
  /// Moves the buffer to the left and reads new data to fill the right part
  bool moveAndFillBuffer();
  /// Increases the size of the buffer and reads new data to fill it
  bool resizeAndFillBuffer();
  /// Moves buffer_pos until the first valid char is not in delim string
  bool trim(const char *delim);
  
  /// Forces to read from the file in the next getToken
  void setBufferAsFull() {
    buffer_pos = max_buffer_len;
    buffer_len = max_buffer_len;
  }

public:
  
  /// Constructor for memory read/write, it receives the string path and the mode
  BufferedMemory(const char *path, const char *mode);
  ~BufferedMemory();
  
  /// Closes the memory
  void close();
  /// Forces to write pending data
  void flush();
  /// Moves the file cursor to the given offset from given whence position
  int seek(int whence, int offset);
  
  /** Lua methods: is more efficient to directly put strings at the Lua stack,
      instead of build a C string and return it. Pushing to the stack
      approximately doubles the memory because needs the original object and the
      Lua string. Returning a C string needs a peak of three times more memory,
      because it needs the original object, the C string, and finally the Lua
      string, even if the C string is removed after written to Lua. **/
  /// Reads a number from the memory and pushes it to the Lua stack
  int readAndPushNumberToLua(lua_State *L);
  //  int readAndPushCharToLua(lua_State *L);
  /// Reads a string of maximum given size from the memory and pushes it to the Lua stack
  int readAndPushStringToLua(lua_State *L, int size);
  /// Reads a line from the memory and pushes it to the Lua stack
  int readAndPushLineToLua(lua_State *L);
  /// Reads the whole memory and pushes it to the Lua stack
  int readAndPushAllToLua(lua_State *L);
  /*****************/
  
  /** For matrix Read/Write C++ interface **/

  // char getChar();
  // be careful, this method returns a pointer to internal buffer

  /// Returns a string of the given maximum size. Be careful, the string is
  /// pointing to the internal buffer, so if you need to perform a copy if you
  /// need it. Otherwise, the string may be destroyed with the next read.
  constString getToken(int size);
  // be careful, this method returns a pointer to internal buffer
  /// Returns a string delimited by any char of the given string. Be careful,
  /// the string is pointing to the internal buffer, so if you need to perform a
  /// copy if you need it. Otherwise, the string may be destroyed with the next
  /// read.
  constString getToken(const char *delim);
  // be careful, this method returns a pointer to internal buffer
  /// Returns a whole line of the file (a string delimited by \n). Be careful,
  /// the string is pointing to the internal buffer, so if you need to perform a
  /// copy if you need it. Otherwise, the string may be destroyed with the next
  /// read.
  constString extract_line();
  // be careful, this method returns a pointer to internal buffer
  /// Returns a whole line of the file (a string delimited by \n), but avoiding
  /// lines which begins with #. Lines beginning with # are taken as commented
  /// lines. Be careful, the string is pointing to the internal buffer, so if
  /// you need to perform a copy if you need it. Otherwise, the string may be
  /// destroyed with the next read.
  constString extract_u_line();
  
  /// Writes a set of values following the given format. Equals to C printf.
  void printf(const char *format, ...);
  
  /// Writes a buffer of chars, given its length, similar to fwrite
  void write(const void *buffer, size_t len);
  
  /// Some objects needs to know the expected size before begin to write things,
  /// so this method is where this size is given.
  void setExpectedSize(int sz) const { }
  
  /// Returns the value of the counter of read/written bytes.
  int getTotalBytes() const { return total_bytes; }
  /**************************************/
};

// Includes the implementation
#include "buffered_memory.impl.h"

#endif // BUFFERED_MEMORY_H
