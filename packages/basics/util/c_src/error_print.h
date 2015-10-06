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

bool getSilentErrorsValue();
void setSilentErrorsValue(bool value);
void print_CPP_LUA_stacktrace_and_exit(int errorcode, const char *format, ...);
void print_CPP_stacktrace(FILE *out = stderr);

#define ERROR_PRINT(strformat)						\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__)

#define ERROR_PRINT1(strformat, v1)					\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__,(v1))

#define ERROR_PRINT2(strformat, v1, v2)					\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2))

#define ERROR_PRINT3(strformat, v1, v2, v3)				\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3))

#define ERROR_PRINT4(strformat, v1, v2, v3, v4)				\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4))

#define ERROR_PRINT5(strformat, v1, v2, v3, v4, v5)			\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4), (v5))

#define ERROR_PRINT6(strformat, v1, v2, v3, v4, v5, v6)			\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4), (v5), (v6))

#define ERROR_PRINT7(strformat, v1, v2, v3, v4, v5, v6, v7)		\
  if (!getSilentErrorsValue()) fprintf(stderr, "Error in file %s in line %d, function %s: "  strformat, __FILE__, __LINE__, __FUNCTION__, (v1), (v2), (v3), (v4), (v5), (v6), (v7))

#define ERROR_EXIT(errorcode, strformat)			\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT1(errorcode, strformat, v1)			\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1)); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT2(errorcode, strformat, v1, v2)			\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1),(v2)); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT3(errorcode, strformat, v1, v2, v3)			\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1),(v2),(v3)); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT4(errorcode, strformat, v1, v2, v3, v4)		\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1),(v2),(v3),(v4)); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT5(errorcode, strformat, v1, v2, v3, v4, v5)		\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1),(v2),(v3),(v4),(v5)); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT6(errorcode, strformat, v1, v2, v3, v4, v5, v6)	\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1),(v2),(v3),(v4),(v5),(v6)); /*exit(errorcode);*/ } while(0)

#define ERROR_EXIT7(errorcode, strformat, v1, v2, v3, v4, v5, v6, v7)	\
  do { if (!getSilentErrorsValue()) print_CPP_LUA_stacktrace_and_exit(errorcode,"Error in file %s in line %d, function %s: " strformat,__FILE__,__LINE__,__FUNCTION__,strformat,(v1),(v2),(v3),(v4),(v5),(v6),(v7)); /*exit(errorcode);*/ } while(0)

#endif // ERROR_EXIT_H
