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
#ifndef SIGNAL_HANDLER_H
#define SIGNAL_HANDLER_H

#include <csignal>
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

#define MAX_SIGNALS 32

namespace april_utils {
  /// This class is not thread-safe, so don't use it with threads.
  class SignalHandler {
    static lua_State *globalL;
    // hash  ( signal -> lua reference )
    static int signal_handlers[MAX_SIGNALS];
    static void sig_handler(int sgn);
    public:
    static void initialize(lua_State *L);
    static void register_signal(int sgn);
    static void release_signal(int sgn);
    static void mask_signals(int how, sigset_t *set);
  };
}

#endif
