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
#include "cfile_stream.h"
#include "lua_file.h"
#include "stream.h"
#include "buffer_stream.h"
#include "lua_buffer_stream.h"
#include "cfile_stream.h"

using namespace april_io;

int lua_isAuxStream(lua_State *L, int index);
Stream *lua_toAuxStream(lua_State *L, int index);
//BIND_END

//BIND_FOOTER_H
template<typename T>
int callLuaFileConstructor(lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  Stream *stream = new T(path, mode);
  if (stream->isOpened()) {
    LuaFile *obj = new LuaFile(stream);
    lua_pushLuaFile(L, obj);
  }
  else {
    delete stream;
    lua_pushnil(L);
  }
  return 1;
}

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
int lua_isAuxStream(lua_State *L, int index) {
  luaL_Stream *p = ((luaL_Stream *)luaL_testudata(L, index, LUA_FILEHANDLE));
  if (p == 0) {
    Stream *s = lua_toStream(L, index);
    if (s == 0) return 0;
  }
  return 1;
}

Stream *lua_toAuxStream(lua_State *L, int index) {
  Stream *s;
  luaL_Stream *p = ((luaL_Stream *)luaL_testudata(L, index, LUA_FILEHANDLE));
  if (p == 0) s = lua_toStream(L, index);
  else s = new CFileStream(fileno(p->f));
  return s;
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

//BIND_METHOD Stream is_opened
{
  LUABIND_RETURN(boolean, obj->isOpened());
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME CFileStream april_io.stream.c_file
//BIND_CPP_CLASS CFileStream
//BIND_SUBCLASS_OF CFileStream Stream

//BIND_CONSTRUCTOR CFileStream
{
  LUABIND_INCREASE_NUM_RETURNS(callFileStreamConstructor<CFileStream>(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME LuaOutputBufferStream april_io.stream.output_lua_buffer
//BIND_CPP_CLASS LuaOutputBufferStream
//BIND_SUBCLASS_OF LuaOutputBufferStream Stream

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
