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
#include "threadable.h"
#include "sig_int_handler.h"

namespace april_thread_utils {
  bool SigIntHandler::installed = false;
  list<Threadable *> SigIntHandler::objects;

  void SigIntHandler::sig_int_handler(int sgn) {
    fprintf (stderr, "\tReceived ctrl+c, stopping %u threads\n",
	     (unsigned int)objects.size());
    for (list<Threadable*>::iterator it = objects.begin();
	 it != objects.end();
	 ++it) {
      fprintf (stderr, "\t\t%s\n", (*it)->getName());
      (*it)->stopThread();
    }
  }
  void SigIntHandler::add_thread(Threadable *obj) {
    objects.push_back(obj);
  }
  void SigIntHandler::remove_thread(Threadable *obj) {
    objects.remove(obj);
  }
  void SigIntHandler::install_handler() {
    if (!installed) {
      installed = true;
      signal(SIGINT, sig_int_handler);
    }
  }
}
