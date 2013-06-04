/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Francisco Zamora-Martinez
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
#ifndef SIG_INT_HANDLER_H
#define SIG_INT_HANDLER_H
#include <cstdio>
#include <csignal>
#include "list.h"
using april_utils::list;

// forward declaration
class Threadable;

namespace april_thread_utils {

  class SigIntHandler {
    static bool installed;
    static list<Threadable *> objects;
    
  public:
    static void sig_int_handler(int sgn);
    static void add_thread(Threadable *obj);
    static void remove_thread(Threadable *obj);
    static void install_handler();
  };

}

#endif // SIG_INT_HANDLER
