/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef ERROR_PRINT_H
#define ERROR_PRINT_H

#include <cstdio>
#include <cstdlib>
extern "C" {
#include "lua.h"
}

void errorPrintSetLuaState(lua_State *L);
void print_CPP_LUA_stacktrace_and_exit(int errorcode);
void print_CPP_stacktrace(FILE *out = stderr);

#define ERROR_PRINT(strformat)						\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__)

#define ERROR_PRINT1(strformat, v1)					\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__,(v1))

#define ERROR_PRINT2(strformat, v1, v2)					\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2))

#define ERROR_PRINT3(strformat, v1, v2, v3)				\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3))

#define ERROR_PRINT4(strformat, v1, v2, v3, v4)				\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4))

#define ERROR_PRINT5(strformat, v1, v2, v3, v4, v5)			\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4), (v5))

#define ERROR_PRINT6(strformat, v1, v2, v3, v4, v5, v6)			\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4), (v5), (v6))

#define ERROR_PRINT7(strformat, v1, v2, v3, v4, v5, v6, v7)		\
  fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4), (v5), (v6), (v7))

#define ERROR_EXIT(errorcode, strformat)			\
  do { ERROR_PRINT(strformat); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT1(errorcode, strformat, v1)			\
  do { ERROR_PRINT1(strformat,(v1)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT2(errorcode, strformat, v1, v2)			\
  do { ERROR_PRINT2(strformat,(v1),(v2)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT3(errorcode, strformat, v1, v2, v3)			\
  do { ERROR_PRINT3(strformat,(v1),(v2),(v3)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT4(errorcode, strformat, v1, v2, v3, v4)		\
  do { ERROR_PRINT4(strformat,(v1),(v2),(v3),(v4)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT5(errorcode, strformat, v1, v2, v3, v4, v5)		\
  do { ERROR_PRINT5(strformat,(v1),(v2),(v3),(v4),(v5)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT6(errorcode, strformat, v1, v2, v3, v4, v5, v6)	\
  do { ERROR_PRINT6(strformat,(v1),(v2),(v3),(v4),(v5),(v6)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT7(errorcode, strformat, v1, v2, v3, v4, v5, v6, v7)	\
  do { ERROR_PRINT7(strformat,(v1),(v2),(v3),(v4),(v5),(v6),(v7)); print_CPP_LUA_stacktrace_and_exit(errorcode); /*exit(errorcode);*/ } while(0)

#endif // ERROR_EXIT_H
