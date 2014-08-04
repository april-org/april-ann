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

//BIND_HEADER_H
#include "stream.h"
#include "stream_buffer.h"
#include "stream_memory.h"
#include "file_stream.h"

using namespace april_io;

int lua_isAuxStream(lua_State *L, int index);
Stream *lua_toAuxStream(lua_State *L, int index);
//BIND_END

//BIND_FOOTER_H
template<typename T>
int callFileStreamConstructor(lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  Stream *stream = new T(path, mode);
  if (stream->isOpened()) {
    lua_pushStream(L, stream);
  }
  else {
    delete stream;
    lua_pushnil(L);
  }
  return 1;
}
//BIND_END

//BIND_HEADER_C
template<typename T = Stream>
int lua_isAuxStream(lua_State *L, int index) {
  luaL_Stream *p = ((luaL_Stream *)luaL_testudata(L, index, LUA_FILEHANDLE));
  if (p == 0) {
    Stream *s = lua_toStream(L, index);
    if (s == 0) return 0;
    T *s_casted = dynamic_cast<T*>(s);
    if (s_casted == 0) return 0;
  }
  return 1;
}

template<typename T = Stream>
T *lua_toAuxStream(lua_State *L, int index) {
  Stream *s;
  luaL_Stream *p = ((luaL_Stream *)luaL_testudata(L, index, LUA_FILEHANDLE));
  if (p == 0) s = lua_toStream(L, index);
  else s = new FileStream(fileno(p->f));
  T *s_casted = dynamic_cast<T*>(s);
  if (s_casted == 0) return 0;
  return s_casted;
}

namespace april_io {
  int readAndPushNumberToLua(lua_State *L, Stream *obj,
                             OutputCStringStream *&c_string) {
    if (c_string == 0) c_string = new OutputCStringStream();
    c_string->clear();
    obj->get(c_string, " ,;\t\n\r");
    if (c_string->empty()) return 0;
    double number;
    if (!c_string->getConstString().extract_double(&number)) {
      ERROR_EXIT(256, "Impossible to extract a number from current file pos\n");
    }
    lua_pushnumber(L, number);
    return 1;
  }
  
  int readAndPushStringToLua(lua_State *L, int size,
                             OutputLuaStringStream *lua_string) {
    if (lua_string == 0) lua_string = new OutputLuaStringStream();
    lua_string->clear();
    obj->get(lua_string, " ,;\t\n\r");
    if (lua_string->empty()) return 0;
    return lua_string->push(L);
  }
  
  int readAndPushLineToLua(lua_State *L, Stream *obj,
                           OutputLuaStringStream *lua_string) {
    if (lua_string == 0) lua_string = new OutputLuaStringStream();
    lua_string->clear();
    extractLineFromStream(obj, lua_string);
    if (lua_string->empty()) return 0;
    return lua_string->push(L);
  }
  
  int readAndPushAllToLua(lua_State *L, Stream *obj,
                          OutputLuaStringStream *lua_string) {
    if (lua_string == 0) lua_string = new OutputLuaStringStream();
    lua_string->clear();
    obj->get(lua_string);
    if (lua_string->empty()) return 0;
    return lua_string->push(L);
  }
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME Stream april_io.stream
//BIND_CPP_CLASS Stream

//BIND_CONSTRUCTOR Stream
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD Stream set_expected_size
{
  size_t sz;
  LUABIND_GET_PARAMETER(1, uint, sz);
  obj->setExpectedSize(sz);
}
//BIND_END

//BIND_METHOD Stream good
{
  LUABIND_RETURN(boolean, obj->good());
}
//BIND_END

//BIND_METHOD Stream eof
{
  LUABIND_RETURN(boolean, obj->eof());
}
//BIND_END

//BIND_METHOD Stream is_opened
{
  LUABIND_RETURN(boolean, obj->isOpened());
}
//BIND_END

//BIND_METHOD Stream close
{
  LUABIND_RETURN(boolean, obj->close());
}
//BIND_END

//BIND_METHOD Stream read
{
  if (!good()) {
    lua_pushnil(L);
    return 1;
  }
  /*
    "*n" reads a number; this is the only format that returns a number instead
    of a string.
      
    "*a" reads the whole file, starting at the current position. On end of
    file, it returns the empty string.
      
    "*l" reads the next line (skipping the end of line), returning nil on end
    of file. This is the default format.  number reads a string with up to
    that number of characters, returning nil on end of file. If number is
    zero, it reads nothing and returns an empty string, or nil on end of file.
  */
  int argn = lua_gettop(L); // number of arguments
  OutputLuaStringStream *lua_string = 0;
  OutputCStringStream *c_string = 0;
  if (argn == 0) {
    LUABIND_INCREASE_NUM_RETURNS(readAndPushLineToLua(L, obj, lua_string))
  }
  else {
    for (int i=1; i<=argn; ++i) {
      if (lua_isnumber(L, i)) {
        int size = luaL_checkint(L, i);
        LUABIND_INCREASE_NUM_RETURNS(readAndPushStringToLua(L, obj, size,
                                                            lua_string));
      }
      else {
        const char *format = luaL_checkstring(L, i);
        // a number
        if (strcmp(format, "*n") == 0) {
          LUABIND_INCREASE_NUM_RETURNS(readAndPushNumberToLua(L, obj,
                                                              c_string));
        }
        // the whole file
        else if (strcmp(format, "*a") == 0) {
          LUABIND_INCREASE_NUM_RETURNS(readAndPushAllToLua(L, obj,
                                                           lua_string));
        }
        // a line
        else if (strcmp(format, "*l") == 0) {
          LUABIND_INCREASE_NUM_RETURNS(readAndPushLineToLua(L, obj, lua_string))
        }
        else {
          LUABIND_FERROR1("Unrecognized format string '%s'", format);
        } // if (strcmp(format), ...) ...
      } // if isnil ... else if isnumber ... else ...
    } // for (int i=1; i<= argn; ++i)
  } // if (argn == 0) ... else
  if (lua_string != 0) DecRef(lua_string);
  if (c_string != 0) DecRef(c_string);
}
//BIND_END

//BIND_METHOD Stream write
{
  int argn = lua_gettop(L); // number of arguments
  for (int i=1; i<=argn; ++i) {
    const char *value = luaL_checkstring(L, i);
    obj->printf("%s", value);
  }
}
//BIND_END

//BIND_METHOD Stream seek
{
  const char *whence = luaL_optstring(L, 1, "cur");
  int offset = luaL_optint(L, 2, 0);
  int int_whence;
  if (strcmp(whence, "cur") == 0) {
    int_whence = SEEK_CUR;
  }
  else if (strcmp(whence, "set") == 0) {
    int_whence = SEEK_SET;
  }
  else {
    int_whence = SEEK_END;
    LUABIND_FERROR1("Not supported whence '%s'", whence);
  }
  int ret = obj->seek(int_whence, offset);
  if (ret < 0) {
    LUABIND_RETURN_NIL(L);
    LUABIND_RETURN(string, "Impossible to execute seek");
  }
  else {
    LUABIND_RETURN(int, ret);
    return 1;
  }
}
//BIND_END

//BIND_METHOD Stream flush
{
  obj->flush();
}
//BIND_END

//BIND_METHOD Stream setvbuf
{
  LUABIND_ERROR("NOT IMPLEMENTED");
}
//BIND_END

//BIND_METHOD Stream has_error
{
  LUABIND_RETURN(boolean, obj->hasError());
}
//BIND_END

//BIND_METHOD Stream error_msg
{
  LUABIND_RETURN(string, obj->getErrorMsg());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FileStream april_io.stream.file
//BIND_CPP_CLASS FileStream
//BIND_SUBCLASS_OF FileStream Stream

//BIND_CONSTRUCTOR FileStream
{
  LUABIND_INCREASE_NUM_RETURNS(callFileStreamConstructor<CFileStream>(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StreamMemory april_io.stream.memory
//BIND_CPP_CLASS StreamMemory
//BIND_SUBCLASS_OF StreamMemory Stream

//BIND_CONSTRUCTOR StreamMemory
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD StreamMemory value
{
  LUABIND_INCREASE_NUM_RETURNS(obj->push(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME OutputLuaStringStream april_io.stream.output_lua_string
//BIND_CPP_CLASS OutputLuaStringStream
//BIND_SUBCLASS_OF OutputLuaStringStream Stream

//BIND_CONSTRUCTOR LuaOutputBufferStream
{
  obj = new LuaOutputBufferStream(L);
  LUABIND_RETURN(LuaOutputBufferStream, obj);
}
//BIND_END

//BIND_METHOD LuaOutputBufferStream value
{
  LUABIND_INCREASE_NUM_RETURNS(obj->push());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME LuaInputBufferStream april_io.stream.input_lua_buffer
//BIND_CPP_CLASS LuaInputBufferStream
//BIND_SUBCLASS_OF LuaInputBufferStream Stream

//BIND_CONSTRUCTOR LuaInputBufferStream
{
  obj = new LuaInputBufferStream(L);
  LUABIND_RETURN(LuaInputBufferStream, obj);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME OutputBufferStream april_io.stream.output_c_buffer
//BIND_CPP_CLASS OutputBufferStream
//BIND_SUBCLASS_OF OutputBufferStream Stream

//BIND_CONSTRUCTOR OutputBufferStream
{
  obj = new OutputBufferStream();
  LUABIND_RETURN(OutputBufferStream, obj);
}
//BIND_END

//BIND_METHOD OutputBufferStream value
{
  constString c = obj->get();
  const char *c_str = c;
  lua_pushlstring(L, c_str, c.len());
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME LuaFile april_io.file
//BIND_CPP_CLASS LuaFile

//BIND_CONSTRUCTOR LuaFile
{
  Stream *stream;
  LUABIND_GET_PARAMETER(1, AuxStream, stream);
  obj = new LuaFile(stream);
  LUABIND_RETURN(LuaFile, obj);
}
//BIND_END

//BIND_METHOD LuaFile read
{
  LUABIND_INCREASE_NUM_RETURNS(obj->readLua(L));
}
//BIND_END

//BIND_METHOD LuaFile write
{
  LUABIND_INCREASE_NUM_RETURNS(obj->writeLua(L));
}
//BIND_END

//BIND_METHOD LuaFile close
{
  LUABIND_INCREASE_NUM_RETURNS(obj->closeLua(L));
}
//BIND_END

//BIND_METHOD LuaFile flush
{
  LUABIND_INCREASE_NUM_RETURNS(obj->flushLua(L));
}
//BIND_END

//BIND_METHOD LuaFile seek
{
  LUABIND_INCREASE_NUM_RETURNS(obj->seekLua(L));
}
//BIND_END

//BIND_METHOD LuaFile good
{
  LUABIND_INCREASE_NUM_RETURNS(obj->goodLua(L));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////
