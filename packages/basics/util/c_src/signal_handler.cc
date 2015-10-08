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
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include "error_print.h"
#include "mystring.h"
#include "signal_handler.h"
#include "unused_variable.h"

/// Table which contains the Lua function references
#define TABLE_NAME "april_signal"

namespace AprilUtils {
  /// The mapping between signals and Lua function references
  int SignalHandler::signal_handlers[MAX_SIGNALS];
  lua_State *SignalHandler::globalL = 0;
  
  /// The handler in C++, a static function which calls the Lua function related
  /// with the given signal number
  void SignalHandler::sig_handler(int sgn) {
    if (globalL == 0) ERROR_EXIT(1024, "Lua state not defined\n");
    if (signal_handlers[sgn] == LUA_REFNIL) {
      ERROR_EXIT(128, "It couldn't be happening... :S\n");
    }
    lua_getfield(globalL, LUA_REGISTRYINDEX, TABLE_NAME);
    lua_rawgeti(globalL, -1, signal_handlers[sgn]);
    if (lua_pcall(globalL, 0, 0, 0) != LUA_OK) {
      AprilUtils::string str(lua_tostring(globalL,-1));
      lua_pop(globalL,1);
      ERROR_EXIT1(128, "%s", str.c_str());
    }
    lua_pop(globalL, 1);
  }

  /// Initialization of all the mappings to LUA_REFNIL, captures the lua_State,
  /// and register of TABLE_NAME in the registry of Lua
  void SignalHandler::initialize(lua_State *L) {
    if (globalL != 0) {
      ERROR_EXIT(256, "Trying to initialize twice\n");
    }
    globalL = L;
    for (int i=0; i<MAX_SIGNALS; ++i) signal_handlers[i] = LUA_REFNIL;
    lua_newtable(globalL);
    lua_setfield(globalL, LUA_REGISTRYINDEX, TABLE_NAME);
  }
  
  /// Registers a Lua function (which is on the top of the stack) to be executed
  /// when signal sgn arrives
  void SignalHandler::register_signal(int sgn) {
    if (globalL == 0) ERROR_EXIT(1024, "Lua state not defined\n");
    if (sgn < 0 || sgn >= MAX_SIGNALS) {
      ERROR_EXIT2(256,"Unknown signal number, found %d, expected to be in "
		  "range [0,%d]\n", sgn, MAX_SIGNALS-1);
    }
    if (signal_handlers[sgn] != LUA_REFNIL) {
      ERROR_PRINT("Trying to register previously registered signal (it will be overwritten)\n");
      release_signal(sgn);
    }
    sig_t handler_func;
    int ref;
    if (lua_isnil(globalL,-1)) {
      ref = LUA_REFNIL;
      handler_func = SIG_IGN;
    }
    else {
      lua_getfield(globalL, LUA_REGISTRYINDEX, TABLE_NAME);
      lua_pushvalue(globalL, -2);
      ref = luaL_ref(globalL, -2);
      lua_pop(globalL, -1);
      handler_func = SignalHandler::sig_handler;
    }
    signal_handlers[sgn] = ref;
    struct sigaction current, old;
    sigset_t mask;
    sigemptyset(&mask);
    current.sa_handler = handler_func;
    current.sa_mask = mask;
    current.sa_flags = 0;
    int ret = sigaction(sgn, &current, &old);
    UNUSED_VARIABLE(old);
    if (ret < 0) {
      ERROR_EXIT1(128, "%s\n", strerror(errno));
    }
    /* TODO:
      if ( ((old.sa_flags & SA_SIGINFO) && old.sa_sigaction != 0) ||
      (!(old.sa_flags & SA_SIGINFO) && old.sa_handler != SIG_IGN && old.sa_handler != SIG_DFL) ) {
      ERROR_PRINT("Registering a signal which was previously "
      "registered by a third (it will be overwritten)\n");
      }
    */
  }

  /// Releases the function associated with the given signal
  void SignalHandler::release_signal(int sgn) {
    if (globalL == 0) ERROR_EXIT(1024, "Lua state not defined\n");
    if (sgn < 0 || sgn >= MAX_SIGNALS) {
      ERROR_EXIT2(256,"Unknown signal number, found %d, expected to be in "
		  "range [0,%d]\n", sgn, MAX_SIGNALS-1);
    }
    struct sigaction current, old;
    sigset_t mask;
    sigemptyset(&mask);
    current.sa_handler = SIG_DFL;
    current.sa_mask = mask;
    current.sa_flags = 0;
    int ret = sigaction(sgn, &current, &old);
    UNUSED_VARIABLE(old);
    if (ret < 0) {
      ERROR_EXIT1(128, "%s\n", strerror(errno));
    }
    else if (signal_handlers[sgn] == LUA_REFNIL) {
      ERROR_EXIT(256, "Releasing a not registered signal or registered by a third\n");
    }
    lua_getfield(globalL, LUA_REGISTRYINDEX, TABLE_NAME);
    lua_pushvalue(globalL, -2);
    luaL_unref(globalL, -2, signal_handlers[sgn]);
    lua_pop(globalL, -1);
    signal_handlers[sgn] = LUA_REFNIL;
  }
  
  /// Masks the given set of signals
  void SignalHandler::mask_signals(int how, sigset_t *set) {
    if (sigprocmask(how, set, 0) != 0) {
      ERROR_EXIT1(128, "%s\n", strerror(errno));
    }
  }
}
