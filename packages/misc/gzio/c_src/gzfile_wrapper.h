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
  
  void printf(const char *format) {
    total_bytes += gzprintf(f, format);
  }

  template<typename T>
  void printf(const char *format, const T &data) {
    total_bytes += gzprintf(f, format, data);
  }
  
  template<typename T1, typename T2>
  void printf(const char *format, const T1 &data1, const T2 &data2) {
    total_bytes += gzprintf(f, format, data1, data2);
  }
  
  template<typename T1, typename T2, typename T3>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3) {
    total_bytes += gzprintf(f, format, data1, data2, data3);
  }

  template<typename T1, typename T2, typename T3, typename T4>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4);
  }

  template<typename T1, typename T2, typename T3, typename T4, typename T5>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4, const T5 &data5) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4, data5);
  }

  template<typename T1, typename T2, typename T3, typename T4, typename T5,
	   typename T6>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4, const T5 &data5,
	      const T6 &data6) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4, data5,
			    data6);
  }

  template<typename T1, typename T2, typename T3, typename T4, typename T5,
	   typename T6, typename T7>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4, const T5 &data5,
	      const T6 &data6, const T7 &data7) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4, data5,
			    data6, data7);
  }

  template<typename T1, typename T2, typename T3, typename T4, typename T5,
	   typename T6, typename T7, typename T8>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4, const T5 &data5,
	      const T6 &data6, const T7 &data7, const T8 data8) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4, data5,
			    data6, data7, data8);
  }

  template<typename T1, typename T2, typename T3, typename T4, typename T5,
	   typename T6, typename T7, typename T8, typename T9>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4, const T5 &data5,
	      const T6 &data6, const T7 &data7, const T8 data8,
	      const T9 &data9) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4, data5,
			    data6, data7, data8, data9);
  }

  template<typename T1, typename T2, typename T3, typename T4, typename T5,
	   typename T6, typename T7, typename T8, typename T9, typename T10>
  void printf(const char *format, const T1 &data1, const T2 &data2,
	      const T3 &data3, const T4 &data4, const T5 &data5,
	      const T6 &data6, const T7 &data7, const T8 data8,
	      const T9 &data9, const T10 &data10) {
    total_bytes += gzprintf(f, format, data1, data2, data3, data4, data5,
			    data6, data7, data8, data9, data10);
  }
  
  void setExpectedSize(int sz) const { }
  
  int getTotalBytes() const { return total_bytes; }
  /**************************************/
};

#endif // GZFILE_WRAPPER_H

