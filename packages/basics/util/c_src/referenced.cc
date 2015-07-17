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

// #define __PRINT_STACK__
// #define __DEBUG__
// #define _debugrefsno0_

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

#include "base.h"
#include "referenced.h"
#ifdef __DEBUG__
#include <cstdio>
#include "error_print.h"
#endif
#ifdef _debugrefsno0_
#include <cstdio>
#endif

Referenced::Referenced() {
#ifdef __DEBUG__
  fprintf(stderr," DEBUG Creating %p\n",this);
#ifdef __PRINT_STACK__
  print_CPP_stacktrace();
#endif
#endif
  refs = 0;
}
Referenced::~Referenced() {
#ifdef __DEBUG__
  fprintf(stderr," DEBUG Destroying %p with reference %d\n",this,refs);
#ifdef __PRINT_STACK__
  print_CPP_stacktrace();
#endif
#endif
#ifdef _debugrefsno0_
  if (refs != 0) {
    fprintf(stderr,"Warning: destroying %p with reference %d!=0\n",this,refs);
  }
#endif
}
void Referenced::incRef() { 
  refs++; 
#ifdef __DEBUG__
  fprintf(stderr," DEBUG IncRef %p to reference %d\n",this,refs);
#ifdef __PRINT_STACK__
  print_CPP_stacktrace();
#endif
#endif
}
bool Referenced::decRef() { 
  refs--;
#ifdef __DEBUG__
  fprintf(stderr," DEBUG DecRef %p to reference %d\n",this,refs);
#ifdef __PRINT_STACK__
  print_CPP_stacktrace();
#endif
#endif
  return (refs <= 0); 
}

int Referenced::getRef() const {
  return refs;
}
