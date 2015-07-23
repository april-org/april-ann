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
#include <cstdlib>
#include <cstdio>
#include "base.h"

namespace Base {
  static lua_State *globalL = 0;
  
  void registerGlobalLuaState(lua_State *L) {
    if (globalL != 0 && globalL != L) {
      fprintf(stderr, "FATAL ERROR: Global Lua State set twice!!!\n");
      exit(128);
    }
    globalL = L;
  }
  
  lua_State *getGlobalLuaState() {
    return globalL;
  }

  lua_State *getAndCheckGlobalLuaState() {
    if (globalL == 0) {
      fprintf(stderr, "FATAL ERROR: Global Lua State not initialized!!!\n");
      exit(128);
    }
    return globalL;
  }
}
