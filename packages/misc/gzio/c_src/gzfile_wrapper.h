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
#ifndef GZFILE_WRAPPER_H
#define GZFILE_WRAPPER_H

#include <zlib.h>
#include <cstring>
#include "referenced.h"
#include "constString.h"
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

class GZFileWrapper : public Referenced {
  int total_bytes;
  char *buffer;
  int max_buffer_len, buffer_pos, buffer_len;
  gzFile f;
  
  bool moveAndFillBuffer();
  bool resizeAndFillBuffer();
  bool trim(const char *delim);
  
  void setBufferAsFull() {
    // forces to read from the file in the next getToken
    buffer_pos = max_buffer_len;
    buffer_len = max_buffer_len;
  }

public:

  static bool isGZ(const char *filename) {
    int len = strlen(filename);
    return (len > 3 && filename[len-3] == '.' &&
	    filename[len-2] == 'g' &&
	    filename[len-1] == 'z');
  }
  
  GZFileWrapper(const char *path, const char *mode);
  ~GZFileWrapper();
  
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

#endif // GZFILE_WRAPPER_H

