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
//BIND_HEADER_C
//BIND_END

//BIND_HEADER_H
#include "gzfile_wrapper.h"
//BIND_END

//BIND_LUACLASSNAME GZFileWrapper gzio
//BIND_CPP_CLASS    GZFileWrapper

//BIND_CONSTRUCTOR GZFileWrapper
{
  LUABIND_ERROR("Use open class method instead of the constructor");
}
//BIND_END

//BIND_CLASS_METHOD GZFileWrapper open
{
  const char *path, *mode;
  LUABIND_GET_PARAMETER(1, string, path);
  LUABIND_GET_OPTIONAL_PARAMETER(2, string, mode, "r");
  GZFileWrapper *obj = new GZFileWrapper(path, mode);
  LUABIND_RETURN(GZFileWrapper, obj);
}
//BIND_END

//BIND_METHOD GZFileWrapper close
{
  obj->close();
}
//BIND_END

//BIND_METHOD GZFileWrapper flush
{
  obj->flush();
}
//BIND_END

//BIND_METHOD GZFileWrapper seek
{
  int offset;
  const char *whence;
  int int_whence;
  LUABIND_GET_OPTIONAL_PARAMETER(1, string, whence, "cur");
  LUABIND_GET_OPTIONAL_PARAMETER(2, int, offset, 0);
  if      (strcmp(whence, "cur") == 0) int_whence = SEEK_CUR;
  else if (strcmp(whence, "set") == 0) int_whence = SEEK_SET;
  else {
    int_whence = SEEK_END;
    LUABIND_FERROR1("Not supported whence '%s'", whence);
  }
  int ret = obj->seek(int_whence, offset);
  if (ret < 0) {
    LUABIND_RETURN_NIL();
    LUABIND_RETURN(string, "Impossible to execute seek");
  }
  else LUABIND_RETURN(int, ret);
}
//BIND_END

//BIND_METHOD GZFileWrapper read
{
  /*
    "*n" reads a number; this is the only format that returns a number instead
    of a string.

    "*a" reads the whole file, starting at the current position. On end of file,
    it returns the empty string.

    "*l" reads the next line (skipping the end of line), returning nil on end of
    file. This is the default format.  number reads a string with up to that
    number of characters, returning nil on end of file. If number is zero, it
    reads nothing and returns an empty string, or nil on end of file.
  */
  int argn = lua_gettop(L); // number of arguments
  int num_returned_values=0;
  if (argn == 0)
    obj->readAndPushLineToLua(L);
  else {
    for (int i=1; i<=argn; ++i) {
      if (lua_isnumber(L, i)) {
	int size;
	LUABIND_GET_PARAMETER(i, int, size);
	num_returned_values += obj->readAndPushStringToLua(L, size);
      }
      else {
	const char *format;
	LUABIND_GET_PARAMETER(i, string, format);
	// a number
	if (strcmp(format, "*n") == 0)
	  num_returned_values += obj->readAndPushNumberToLua(L);
	// the whole file
	else if (strcmp(format, "*a") == 0)
	  num_returned_values += obj->readAndPushAllToLua(L);
	// a line
	else if (strcmp(format, "*l") == 0)
	  num_returned_values += obj->readAndPushLineToLua(L);
	else
	  LUABIND_FERROR1("Unrecognized format string '%s'", format);
      }
    }
  }
  // avoid the default return of LUABIND
  return num_returned_values;
}
//BIND_END

//BIND_METHOD GZFileWrapper write
{
  int argn = lua_gettop(L); // number of arguments
  for (int i=1; i<=argn; ++i) {
    const char *value;
    LUABIND_GET_PARAMETER(i, string, value);
    obj->printf("%s", value);
  }
}
//BIND_END
