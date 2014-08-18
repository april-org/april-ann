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
#include "archive_package.h"
#include "buffered_stream.h"
#include "c_string.h"
#include "file_stream.h"
#include "lua_string.h"
#include "serializable.h"
#include "stream.h"
#include "stream_memory.h"

using namespace AprilIO;
//BIND_END

//BIND_FOOTER_H
template<typename T>
int callFileStreamConstructor(lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  StreamInterface *stream = new T(path, mode);
  if (stream->isOpened()) {
    lua_pushStreamInterface(L, stream);
  }
  else {
    lua_pushnil(L);
    if (stream->hasError()) lua_pushstring(L, stream->getErrorMsg());
    delete stream;
  }
  return 1;
}

template<typename T>
int callArchivePackageConstructor(lua_State *L) {
  ArchivePackage *obj;
  if (lua_isstring(L,1)) {
    // from a file name
    const char *path = luaL_checkstring(L, 1);
    const char *mode = luaL_optstring(L, 2, "r");
    obj = new T(path, mode);
  }
  else {
    // from a stream
    StreamInterface *stream = lua_toStreamInterface(L,1);
    obj = new T(stream);
  }
  if (obj->good()) {
    lua_pushArchivePackage(L, obj);
  }
  else {
    lua_pushnil(L);
    delete obj;
  }
  return 1;
}

template<typename T>
int lua_isAuxStreamInterface(lua_State *L, int index) {
  luaL_Stream *p = ((luaL_Stream *)luaL_testudata(L, index, LUA_FILEHANDLE));
  if (p == 0) {
    StreamInterface *s = lua_toStreamInterface(L, index);
    if (s == 0) return 0;
    T *s_casted = dynamic_cast<T*>(s);
    if (s_casted == 0) return 0;
  }
  return 1;
}

template<typename T>
T *lua_toAuxStreamInterface(lua_State *L, int index) {
  StreamInterface *s;
  luaL_Stream *p = ((luaL_Stream *)luaL_testudata(L, index, LUA_FILEHANDLE));
  if (p == 0) s = lua_toStreamInterface(L, index);
  else s = new FileStream(p->f);
  T *s_casted = dynamic_cast<T*>(s);
  if (s_casted == 0) return 0;
  return s_casted;
}
//BIND_END

//BIND_HEADER_C
namespace AprilIO {

  int readAndPushNumberToLua(lua_State *L, StreamInterface *obj,
                             CStringStream *&c_string) {
    if (c_string == 0) AssignRef(c_string, new CStringStream());
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
  
  int readAndPushStringToLua(lua_State *L, StreamInterface *obj, int size,
                             OutputLuaStringStream *&lua_string) {
    if (lua_string == 0) AssignRef(lua_string, new OutputLuaStringStream(L));
    lua_string->clear();
    obj->get(lua_string, size);
    if (lua_string->empty()) return 0;
    return lua_string->push(L);
  }
  
  int readAndPushLineToLua(lua_State *L, StreamInterface *obj,
                           OutputLuaStringStream *&lua_string) {
    if (lua_string == 0) AssignRef(lua_string, new OutputLuaStringStream(L));
    lua_string->clear();
    extractLineFromStream(obj, lua_string);
    if (lua_string->empty()) return 0;
    return lua_string->push(L);
  }
  
  int readAndPushAllToLua(lua_State *L, StreamInterface *obj,
                          OutputLuaStringStream *&lua_string) {
    if (lua_string == 0) AssignRef(lua_string, new OutputLuaStringStream(L));
    lua_string->clear();
    obj->get(lua_string);
    if (lua_string->empty()) return 0;
    return lua_string->push(L);
  }
}

//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StreamInterface aprilio.stream
//BIND_CPP_CLASS StreamInterface

//BIND_CONSTRUCTOR StreamInterface
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD StreamInterface good
{
  LUABIND_RETURN(boolean, obj->good());
}
//BIND_END

//BIND_METHOD StreamInterface eof
{
  LUABIND_RETURN(boolean, obj->eof());
}
//BIND_END

//BIND_METHOD StreamInterface is_opened
{
  LUABIND_RETURN(boolean, obj->isOpened());
}
//BIND_END

//BIND_METHOD StreamInterface close
{
  obj->close();
}
//BIND_END

//BIND_METHOD StreamInterface read
{
  if (!obj->good()) {
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
  CStringStream *c_string = 0;
  if (argn == 0) {
    LUABIND_INCREASE_NUM_RETURNS(readAndPushLineToLua(L, obj, lua_string));
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
          LUABIND_INCREASE_NUM_RETURNS(readAndPushLineToLua(L, obj, lua_string));
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

//BIND_METHOD StreamInterface write
{
  int argn = lua_gettop(L); // number of arguments
  for (int i=1; i<=argn; ++i) {
    const char *value = luaL_checkstring(L, i);
    obj->put(value, luaL_len(L, i));
  }
  if (obj->hasError()) {
    LUABIND_RETURN_NIL();
    LUABIND_RETURN(string, obj->getErrorMsg());
  }
  else {
    LUABIND_RETURN(StreamInterface, obj);
  }
}
//BIND_END

//BIND_METHOD StreamInterface seek
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
  off_t ret = obj->seek(int_whence, offset);
  if (ret < 0) {
    LUABIND_RETURN_NIL();
    LUABIND_RETURN(string, obj->getErrorMsg());
  }
  else {
    // sanity check
    april_assert( static_cast<off_t>( static_cast<double>( ret ) ) == ret );
    LUABIND_RETURN(number, static_cast<double>(ret));
    return 1;
  }
}
//BIND_END

//BIND_METHOD StreamInterface flush
{
  obj->flush();
}
//BIND_END

//BIND_METHOD StreamInterface setvbuf
{
  LUABIND_ERROR("NOT IMPLEMENTED");
}
//BIND_END

//BIND_METHOD StreamInterface has_error
{
  LUABIND_RETURN(boolean, obj->hasError());
}
//BIND_END

//BIND_METHOD StreamInterface error_msg
{
  LUABIND_RETURN(string, obj->getErrorMsg());
}
//BIND_END

//BIND_METHOD StreamInterface get
{
  april_utils::SharedPtr<StreamInterface> dest;
  OutputLuaStringStream *aux_lua_string = 0;
  size_t size = SIZE_MAX;
  const char *delim = 0;
  bool keep_delim = false;
  if (lua_istable(L,1)) {
    // complete API
    check_table_fields(L, 1, "dest", "size", "delim", "keep_delim",
                       (const char*)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, dest, StreamInterface, dest, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, SIZE_MAX);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, delim, string, delim, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, keep_delim, boolean, keep_delim, false);
  }
  else {
    // simplified API
    if (lua_isuint(L,1)) LUABIND_GET_PARAMETER(1, uint, size);
    else if (lua_isstring(L,1)) LUABIND_GET_PARAMETER(1, string, delim);
    else LUABIND_ERROR("Needs delimitiers string or number as 1st argument");
    LUABIND_GET_OPTIONAL_PARAMETER(2, boolean, keep_delim, false);
  }
  if (dest.empty()) {
    aux_lua_string = new OutputLuaStringStream(L);
    dest = aux_lua_string;
  }
  size_t len = obj->get(dest.get(), size, delim, keep_delim);
  if (len != 0 || !obj->eof()) {
    if (aux_lua_string == 0) LUABIND_RETURN(StreamInterface, dest.get());
    else LUABIND_INCREASE_NUM_RETURNS(aux_lua_string->push(L));
    LUABIND_RETURN(uint, len);
  }
}
//BIND_END

//BIND_METHOD StreamInterface put
{
  april_utils::SharedPtr<StreamInterface> ptr;
  const char *buffer;
  size_t len;
  if (lua_isStreamInterface(L,1)) {
    size_t size;
    LUABIND_GET_PARAMETER(1, StreamInterface, ptr);
    LUABIND_GET_OPTIONAL_PARAMETER(2, uint, size, SIZE_MAX);
    len = obj->put(ptr.get(), size);
  }
  else {
    LUABIND_GET_PARAMETER(1, string, buffer);
    len = obj->put(buffer);
  }
  LUABIND_RETURN(uint, len);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FileStream aprilio.stream.file
//BIND_CPP_CLASS FileStream
//BIND_SUBCLASS_OF FileStream StreamInterface

//BIND_CONSTRUCTOR FileStream
{
  LUABIND_INCREASE_NUM_RETURNS(callFileStreamConstructor<FileStream>(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME InputLuaStringStream aprilio.stream.input_lua_string
//BIND_CPP_CLASS InputLuaStringStream
//BIND_SUBCLASS_OF InputLuaStringStream StreamInterface

//BIND_CONSTRUCTOR InputLuaStringStream
{
  obj = new InputLuaStringStream(L, 1);
  LUABIND_RETURN(InputLuaStringStream, obj);
}
//BIND_END

//BIND_METHOD InputLuaStringStream value
{
  LUABIND_INCREASE_NUM_RETURNS(obj->push(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME OutputLuaStringStream aprilio.stream.output_lua_string
//BIND_CPP_CLASS OutputLuaStringStream
//BIND_SUBCLASS_OF OutputLuaStringStream StreamInterface

//BIND_CONSTRUCTOR OutputLuaStringStream
{
  obj = new OutputLuaStringStream(L, 1);
  LUABIND_RETURN(OutputLuaStringStream, obj);
}
//BIND_END

//BIND_METHOD OutputLuaStringStream value
{
  LUABIND_INCREASE_NUM_RETURNS(obj->push(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME CStringStream aprilio.stream.c_string
//BIND_CPP_CLASS CStringStream
//BIND_SUBCLASS_OF CStringStream StreamInterface

//BIND_CONSTRUCTOR CStringStream
{
  const char *str;
  LUABIND_GET_OPTIONAL_PARAMETER(1, string, str, 0);
  if (str != 0) {
    obj = new CStringStream(str, luaL_len(L,1));
  }
  else {
    obj = new CStringStream();
  }
  LUABIND_RETURN(CStringStream, obj);
}
//BIND_END

//BIND_METHOD CStringStream value
{
  LUABIND_INCREASE_NUM_RETURNS(obj->push(L));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ArchivePackage aprilio.package
//BIND_CPP_CLASS ArchivePackage

//BIND_CONSTRUCTOR ArchivePackage
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD ArchivePackage has_error
{
  LUABIND_RETURN(boolean, obj->hasError());
}
//BIND_END

//BIND_METHOD ArchivePackage error_msg
{
  LUABIND_RETURN(string, obj->getErrorMsg());
}
//BIND_END

//BIND_METHOD ArchivePackage close
{
  obj->close();
}
//BIND_END

//BIND_METHOD ArchivePackage open
{
  int flags = 0;
  StreamInterface *stream;
  int argn = lua_gettop(L); // number of arguments
  for (int i=2; i<=argn; ++i) {
    int current = lua_toint(L,i);
    flags |= current;
  }
  if (lua_isnumber(L,1)) {
    // open with a file index number
    unsigned int idx;
    LUABIND_GET_PARAMETER(1, uint, idx);
    if (idx < 1) LUABIND_ERROR("Index starts at 1");
    stream = obj->openFile(idx - 1, flags);
  }
  else {
    // open with a file name
    const char *filename;
    LUABIND_GET_PARAMETER(1, string, filename);
    stream = obj->openFile(filename, flags);
  }
  if (stream != 0) {
    LUABIND_RETURN(StreamInterface, stream);
  }
  else {
    LUABIND_RETURN_NIL();
    LUABIND_RETURN(string, "Unable to open the file");
  }
}
//BIND_END

//BIND_METHOD ArchivePackage good
{
  LUABIND_RETURN(boolean, obj->good());
}
//BIND_END

//BIND_METHOD ArchivePackage number_of_files
{
  LUABIND_RETURN(uint, obj->getNumberOfFiles());
}
//BIND_END

//BIND_METHOD ArchivePackage name_of
{
  unsigned int idx;
  LUABIND_GET_PARAMETER(1, uint, idx);
  if (idx < 1) LUABIND_ERROR("Index starts at 1");
  const char *name = obj->getNameOf(idx - 1);
  if (name == 0) LUABIND_RETURN_NIL();
  else LUABIND_RETURN(string, name);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_CPP_CLASS Serializable

//BIND_CONSTRUCTOR Serializable
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD Serializable write
{
  OutputLuaStringStream *aux_lua_string;
  StreamInterface *ptr;
  const char *mode;
  april_utils::SharedPtr<StreamInterface> dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, StreamInterface, ptr, 0);
  april_utils::LuaTableOptions options(L,2);
  if (ptr == 0) dest = aux_lua_string = new OutputLuaStringStream(L);
  else dest = ptr;
  obj->write(dest.get(), &options);
  if (ptr == 0) LUABIND_INCREASE_NUM_RETURNS(aux_lua_string->push(L));
  else LUABIND_RETURN(StreamInterface, ptr);
}
//BIND_END
