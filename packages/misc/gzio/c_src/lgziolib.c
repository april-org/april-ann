/*
 * Modified by Francisco Zamora-Martinez
 *
 * Copyright (C) 2013 Francisco Zamora-Martinez
 *
 * Adapted to work with lua 5.2.2
 *
 */

/*
 * gzip file I/O library
 *
 * Copyright (C) 2007 Judge Maygarden
 *
 * This file was created by Judge Maygarden (jmaygarden at computer dot org)
 * through trivial changes to the "liolib.c" file from Lua 5.1.2. It was
 * inspired by a post from David Burgess on the official Lua mailing list
 * (http://www.lua.org/lua-l.html). Use it on gzipped files as you would use the
 * standard Lua "io" module on uncompressed text files.
 */

/******************************************************************************
 * * Copyright (C) 1994-2007 Lua.org, PUC-Rio.  All rights reserved.
 * *
 * * Permission is hereby granted, free of charge, to any person obtaining
 * * a copy of this software and associated documentation files (the
 * * "Software"), to deal in the Software without restriction, including
 * * without limitation the rights to use, copy, modify, merge, publish,
 * * distribute, sublicense, and/or sell copies of the Software, and to
 * * permit persons to whom the Software is furnished to do so, subject to
 * * the following conditions:
 * *
 * * The above copyright notice and this permission notice shall be
 * * included in all copies or substantial portions of the Software.
 * *
 * * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ******************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <zlib.h>

#include "lgziolib.h"

#include "lauxlib.h"
#include "lualib.h"

#define LUA_GZIOLIBNAME   "gzio"
#define LUA_GZFILEHANDLE  "gzFile"

#define IO_INPUT	1
#define IO_OUTPUT	2

static const char *const fnames[] = {"input", "output"};


static int pushresult (lua_State *L, int i, const char *filename) {
  int en = errno;  /* calls to Lua API may change this value */
  if (i) {
    lua_pushboolean(L, 1);
    return 1;
  }
  else {
    lua_pushnil(L);
    if (filename)
      lua_pushfstring(L, "%s: %s", filename, strerror(en));
    else
      lua_pushfstring(L, "%s", strerror(en));
    lua_pushinteger(L, en);
    return 3;
  }
}


static int pushgzresult (lua_State *L, gzFile f, int i, const char *filename) {
  int en = errno;  /* calls to Lua API may change this value */
  int gzen;
  const char *s;
  if (i) {
    lua_pushboolean(L, 1);
    return 1;
  }
  else {
    lua_pushnil(L);
    s = gzerror(f, &gzen);
    if (Z_ERRNO == gzen)
      s = strerror(en);
    if (filename)
      lua_pushfstring(L, "%s: %s", filename, s);
    else
      lua_pushfstring(L, "%s", s);
    lua_pushinteger(L, gzen);
    lua_pushinteger(L, en);
    return 4;
  }
}


static void fileerror (lua_State *L, int arg, const char *filename) {
  lua_pushfstring(L, "%s: %s", filename, strerror(errno));
  luaL_argerror(L, arg, lua_tostring(L, -1));
}


#define topfile(L)	((gzFile *)luaL_checkudata(L, 1, LUA_GZFILEHANDLE))


static int io_type (lua_State *L) {
  void *ud;
  luaL_checkany(L, 1);
  ud = lua_touserdata(L, 1);
  lua_getfield(L, LUA_REGISTRYINDEX, LUA_GZFILEHANDLE);
  if (ud == NULL || !lua_getmetatable(L, 1) || !lua_rawequal(L, -2, -1))
    lua_pushnil(L);  /* not a file */
  else if (*((gzFile *)ud) == NULL)
    lua_pushliteral(L, "closed file");
  else
    lua_pushliteral(L, "file");
  return 1;
}


static gzFile tofile (lua_State *L) {
  gzFile *f = topfile(L);
  if (*f == NULL)
    luaL_error(L, "attempt to use a closed file");
  return *f;
}



/*
** When creating file handles, always creates a `closed' file handle
** before opening the actual file; so, if there is a memory error, the
** file is not left opened.
*/
static gzFile *newfile (lua_State *L) {
  gzFile *pf = (gzFile *)lua_newuserdata(L, sizeof(gzFile));
  *pf = NULL;  /* file handle is currently `closed' */
  luaL_getmetatable(L, LUA_GZFILEHANDLE);
  lua_setmetatable(L, -2);
  return pf;
}


/*
** this function has a separated environment, which defines the
** correct __close for 'popen' files
*/
static int io_pclose (lua_State *L) {
  gzFile *p = topfile(L);
  int ok = lua_pclose(L, *p);
  *p = NULL;
  return pushresult(L, ok, NULL);
}


static int io_fclose (lua_State *L) {
  gzFile *p = topfile(L);
  int ok = (gzclose(*p) == 0);
  *p = NULL;
  return pushresult(L, ok, NULL);
}


static int aux_close (lua_State *L) {
  lua_getuservalue(L, 1);
  // lua_getfenv(L, 1);
  lua_getfield(L, -1, "__close");
  return (lua_tocfunction(L, -1))(L);
}


static int io_close (lua_State *L) {
  if (lua_isnone(L, 1)) {
    lua_getuservalue(L, -1);
    lua_rawgeti(L, -1, IO_OUTPUT);
  }
  tofile(L);  /* make sure argument is a file */
  return aux_close(L);
}


static int io_gc (lua_State *L) {
  gzFile f = *topfile(L);
  /* ignore closed files and standard files */
  if ((void*)f != NULL &&
      (void*)f != (void*)stdin &&
      (void*)f != (void*)stdout &&
      (void*)f != (void*)stderr)
    aux_close(L);
  return 0;
}


static int io_tostring (lua_State *L) {
  gzFile f = *topfile(L);
  if (f == NULL)
    lua_pushstring(L, "file (closed)");
  else
    lua_pushfstring(L, "file (%p)", f);
  return 1;
}


static int io_open (lua_State *L) {
  const char *filename = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  gzFile *pf = newfile(L);
  *pf = gzopen(filename, mode);
  return (*pf == NULL) ? pushresult(L, 0, filename) : 1;
}


static int io_popen (lua_State *L) {
  const char *filename = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  gzFile *pf = newfile(L);
  *pf = (gzFile)lua_popen(L, filename, mode);
  return (*pf == NULL) ? pushresult(L, 0, filename) : 1;
}


static int io_tmpfile (lua_State *L) {
  gzFile *pf = newfile(L);
  *pf = (gzFile)tmpfile();
  return (*pf == NULL) ? pushresult(L, 0, NULL) : 1;
}


static gzFile getiofile (lua_State *L, int findex) {
  gzFile f;
  lua_getuservalue(L, -1);
  lua_rawgeti(L, -1, findex);
  f = *(gzFile *)lua_touserdata(L, -1);
  if (f == NULL)
    luaL_error(L, "standard %s file is closed", fnames[findex - 1]);
  return f;
}


static int g_iofile (lua_State *L, int f, const char *mode) {
  if (!lua_isnoneornil(L, 1)) {
    const char *filename = lua_tostring(L, 1);
    if (filename) {
      gzFile *pf = newfile(L);
      *pf = gzopen(filename, mode);
      if (*pf == NULL)
        fileerror(L, 1, filename);
    }
    else {
      tofile(L);  /* check that it's a valid file handle */
      lua_pushvalue(L, 1);
    }
    lua_rawseti(L, LUA_ENVIRONINDEX, f);
  }
  /* return current value */
  lua_rawgeti(L, LUA_ENVIRONINDEX, f);
  return 1;
}


static int io_input (lua_State *L) {
  return g_iofile(L, IO_INPUT, "r");
}


static int io_output (lua_State *L) {
  return g_iofile(L, IO_OUTPUT, "w");
}


static int io_readline (lua_State *L);


static void aux_lines (lua_State *L, int idx, int toclose) {
  lua_pushvalue(L, idx);
  lua_pushboolean(L, toclose);  /* close/not close file when finished */
  lua_pushcclosure(L, io_readline, 2);
}


static int f_lines (lua_State *L) {
  tofile(L);  /* check that it's a valid file handle */
  aux_lines(L, 1, 0);
  return 1;
}


static int io_lines (lua_State *L) {
  if (lua_isnoneornil(L, 1)) {  /* no arguments? */
    /* will iterate over default input */
    lua_rawgeti(L, LUA_ENVIRONINDEX, IO_INPUT);
    return f_lines(L);
  }
  else {
    const char *filename = luaL_checkstring(L, 1);
    gzFile *pf = newfile(L);
    *pf = gzopen(filename, "r");
    if (*pf == NULL)
      fileerror(L, 1, filename);
    aux_lines(L, lua_gettop(L), 1);
    return 1;
  }
}


/*
** {======================================================
** READ
** =======================================================
*/


static int read_number (lua_State *L, gzFile f) {
  lua_Number d;
  if (fscanf((FILE*)f, LUA_NUMBER_SCAN, &d) == 1) {
    lua_pushnumber(L, d);
    return 1;
  }
  else return 0;  /* read fails */
}


static int test_eof (lua_State *L, gzFile f) {
  lua_pushlstring(L, NULL, 0);
  return (0 != gzeof(f));
}


static int read_line (lua_State *L, gzFile f) {
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  for (;;) {
    size_t l;
    char *p = luaL_prepbuffer(&b);
    if (gzgets(f, p, LUAL_BUFFERSIZE) == NULL) {  /* eof? */
      luaL_pushresult(&b);  /* close buffer */
      return (lua_strlen(L, -1) > 0);  /* check whether read something */
    }
    l = strlen(p);
    if (l == 0 || p[l-1] != '\n')
      luaL_addsize(&b, l);
    else {
      luaL_addsize(&b, l - 1);  /* do not include `eol' */
      luaL_pushresult(&b);  /* close buffer */
      return 1;  /* read at least an `eol' */
    }
  }
}


static int read_chars (lua_State *L, gzFile f, size_t n) {
  size_t rlen;  /* how much to read */
  size_t nr;  /* number of chars actually read */
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  rlen = LUAL_BUFFERSIZE;  /* try to read that much each time */
  do {
    char *p = luaL_prepbuffer(&b);
    if (rlen > n) rlen = n;  /* cannot read more than asked */
    nr = gzread(f, p, rlen);
    luaL_addsize(&b, nr);
    n -= nr;  /* still have to read `n' chars */
  } while (n > 0 && nr == rlen);  /* until end of count or eof */
  luaL_pushresult(&b);  /* close buffer */
  return (n == 0 || lua_strlen(L, -1) > 0);
}


static int g_read (lua_State *L, gzFile f, int first) {
  int nargs = lua_gettop(L) - 1;
  int success;
  int n;
  int errnum;
  if (nargs == 0) {  /* no arguments? */
    success = read_line(L, f);
    n = first+1;  /* to return 1 result */
  }
  else {  /* ensure stack space for all results and for auxlib's buffer */
    luaL_checkstack(L, nargs+LUA_MINSTACK, "too many arguments");
    success = 1;
    for (n = first; nargs-- && success; n++) {
      if (lua_type(L, n) == LUA_TNUMBER) {
        size_t l = (size_t)lua_tointeger(L, n);
        success = (l == 0) ? test_eof(L, f) : read_chars(L, f, l);
      }
      else {
        const char *p = lua_tostring(L, n);
        luaL_argcheck(L, p && p[0] == '*', n, "invalid option");
        switch (p[1]) {
          case 'n':  /* number */
            success = read_number(L, f);
            break;
          case 'l':  /* line */
            success = read_line(L, f);
            break;
          case 'a':  /* file */
            read_chars(L, f, ~((size_t)0));  /* read MAX_SIZE_T chars */
            success = 1; /* always success */
            break;
          default:
            return luaL_argerror(L, n, "invalid format");
        }
      }
    }
  }
  gzerror(f, &errnum);
  if ((Z_OK != errnum) && (Z_STREAM_END != errnum))
    return pushresult(L, 0, NULL);
  if (!success) {
    lua_pop(L, 1);  /* remove last result */
    lua_pushnil(L);  /* push nil instead */
  }
  return n - first;
}


static int io_read (lua_State *L) {
  return g_read(L, getiofile(L, IO_INPUT), 1);
}


static int f_read (lua_State *L) {
  return g_read(L, tofile(L), 2);
}


static int io_readline (lua_State *L) {
  gzFile f = *(gzFile *)lua_touserdata(L, lua_upvalueindex(1));
  const char *errmsg;
  int errnum;
  int sucess;
  if (f == NULL)  /* file is already closed? */
    luaL_error(L, "file is already closed");
  sucess = read_line(L, f);
  errmsg = gzerror(f, &errnum);
  if (Z_OK > errnum)
    return luaL_error(L, "%s", errmsg);
  if (sucess) return 1;
  else {  /* EOF */
    if (lua_toboolean(L, lua_upvalueindex(2))) {  /* generator created file? */
      lua_settop(L, 0);
      lua_pushvalue(L, lua_upvalueindex(1));
      aux_close(L);  /* close it */
    }
    return 0;
  }
}

/* }====================================================== */


static int g_write (lua_State *L, gzFile f, int arg) {
  int nargs = lua_gettop(L) - 1;
  int status = 1;
  for (; nargs--; arg++) {
    if (lua_type(L, arg) == LUA_TNUMBER) {
      /* optimization: could be done exactly as for strings */
      status = status &&
	fprintf((FILE*)f, LUA_NUMBER_FMT, lua_tonumber(L, arg)) > 0;
    }
    else {
      size_t l;
      const char *s = luaL_checklstring(L, arg, &l);
      status = status && ((size_t)gzwrite(f, s, (unsigned) l) == l);
    }
  }
  return pushresult(L, status, NULL);
}


static int io_write (lua_State *L) {
  return g_write(L, getiofile(L, IO_OUTPUT), 1);
}


static int f_write (lua_State *L) {
  return g_write(L, tofile(L), 2);
}


static int f_seek (lua_State *L) {
  static const int mode[] = {SEEK_SET, SEEK_CUR, SEEK_END};
  static const char *const modenames[] = {"set", "cur", "end", NULL};
  gzFile f = tofile(L);
  int op = luaL_checkoption(L, 2, "cur", modenames);
  long offset = luaL_optlong(L, 3, 0);
  op = gzseek(f, offset, mode[op]);
  if (-1 == op)
    return pushgzresult(L, f, 0, NULL);  /* error */
  else {
    lua_pushinteger(L, op);
    return 1;
  }
}


static int f_setvbuf (lua_State *L) {
  static const int mode[] = {_IONBF, _IOFBF, _IOLBF};
  static const char *const modenames[] = {"no", "full", "line", NULL};
  gzFile f = tofile(L);
  int op = luaL_checkoption(L, 2, NULL, modenames);
  lua_Integer sz = luaL_optinteger(L, 3, LUAL_BUFFERSIZE);
  int res = setvbuf((FILE*)f, NULL, mode[op], sz);
  return pushresult(L, res == 0, NULL);
}



static int io_flush (lua_State *L) {
  return pushresult(L, gzflush(getiofile(L, IO_OUTPUT), Z_SYNC_FLUSH) == 0,
	NULL);
}


static int f_flush (lua_State *L) {
  return pushresult(L, gzflush(tofile(L), Z_SYNC_FLUSH) == 0, NULL);
}


static const luaL_Reg iolib[] = {
  {"close", io_close},
  {"flush", io_flush},
  {"input", io_input},
  {"lines", io_lines},
  {"open", io_open},
  {"output", io_output},
  {"popen", io_popen},
  {"read", io_read},
  {"tmpfile", io_tmpfile},
  {"type", io_type},
  {"write", io_write},
  {NULL, NULL}
};


static const luaL_Reg flib[] = {
  {"close", io_close},
  {"flush", f_flush},
  {"lines", f_lines},
  {"read", f_read},
  {"seek", f_seek},
  {"setvbuf", f_setvbuf},
  {"write", f_write},
  {"__gc", io_gc},
  {"__tostring", io_tostring},
  {NULL, NULL}
};


static void createmeta (lua_State *L) {
  /* create metatable for gz file handles */
  luaL_newmetatable(L, LUA_GZFILEHANDLE);
  lua_pushvalue(L, -1);  /* push metatable */
  lua_setfield(L, -2, "__index");  /* metatable.__index = metatable */
  luaL_register(L, NULL, flib);  /* file methods */
}


static void createstdfile (lua_State *L, gzFile f, int k, const char *fname) {
  *newfile(L) = f;
  if (k > 0) {
    lua_pushvalue(L, -1);
    lua_rawseti(L, LUA_ENVIRONINDEX, k);
  }
  lua_setfield(L, -2, fname);
}


GZIO_API int luaopen_gzio (lua_State *L) {
  createmeta(L);

  /* create (private) environment (with fields IO_INPUT, IO_OUTPUT, __close) */
  lua_createtable(L, 2, 1);
  lua_replace(L, LUA_ENVIRONINDEX);

  /* open library */
  luaL_register(L, LUA_GZIOLIBNAME, iolib);

  /* add module metadata */
  lua_pushliteral (L, "_COPYRIGHT");
  lua_pushliteral (L, "Copyright (C) 2007 Judge Maygarden");
  lua_settable (L, -3);
  lua_pushliteral (L, "_DESCRIPTION");
  lua_pushliteral (L, "Lua gzip file I/O module");
  lua_settable (L, -3);
  lua_pushliteral (L, "_VERSION");
  lua_pushliteral (L, "gzio 0.9.0");
  lua_settable (L, -3);

  /* create (and set) default files */
  createstdfile(L, (gzFile)stdin, IO_INPUT, "stdin");
  createstdfile(L, (gzFile)stdout, IO_OUTPUT, "stdout");
  createstdfile(L, (gzFile)stderr, 0, "stderr");

  /* create environment for 'popen' */
  lua_getfield(L, -1, "popen");
  lua_createtable(L, 0, 1);
  lua_pushcfunction(L, io_pclose);
  lua_setfield(L, -2, "__close");
  lua_setuservalue(L, -2);
  // lua_setfenv(L, -2);
  lua_pop(L, 1);  /* pop 'popen' */

  /* set default close function */
  lua_pushcfunction(L, io_fclose);
  lua_setfield(L, LUA_ENVIRONINDEX, "__close");

  return 1;
}
