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
#ifndef THREADBLE_H
#define THREADBLE_H

#include <pthread.h>
#include <cstring>
#include "sig_int_handler.h"
#include "referenced.h"


using AprilThreadUtils::SigIntHandler;

class Threadable : public Referenced {
  // controla la ejecucion del bucle
  bool running;
  // controla si se ha ejecutado el join
  bool joined;
  // el ID del thread
  pthread_t thread_id;
  // nombre
  char *name;
  
  static void *execute(void *ptr) {
    Threadable *obj = reinterpret_cast<Threadable*>(ptr);
    while (obj->running && obj->threadProcedure());
    obj->running = false;
    pthread_exit(0);
  }
  
protected:
  // permite ejecutar cosas ANTES de hacer el running
  virtual void executeBeforeStart() {
  }
  // permite ejecutar cosas ANTES de hacer el stop
  virtual void executeBeforeStop() {
  }
  // permite ejecutar cosas DESPUES de hacer el wait
  virtual void executeAfterWait() {
  }
  // el codigo principal del thread, que va dentro del bucle. Si
  // devuelve false, seria equivalente a hacer un stop del thread
  virtual bool threadProcedure() = 0;
  
public:
  Threadable(const char *name=0) : running(false), joined(true) {
    if (name) {
      this->name = new char[strlen(name)+1];
      strcpy(this->name, name);
    }
    else this->name = 0;
  }
  virtual ~Threadable() {
    stopThread();
    waitThread();
    delete[] name;
  }
  const char *getName() { return name; }
  bool isRunning() { return running; }
  bool isJoined() { return joined; }
  void startThread() {
    if (!running) {
      if (!joined) waitThread();
      joined  = false;
      running = true;
      executeBeforeStart();
      pthread_create(&thread_id, 0, execute, this);
      SigIntHandler::add_thread(this);
    }
  }
  void stopThread() {
    if (running) {
      running = false;
      executeBeforeStop();
      SigIntHandler::remove_thread(this);
      fprintf(stderr, "\t\t\t%s stopped\n", getName());
    }
  }
  void waitThread() {
    if (!running && !joined) {
      pthread_join(thread_id, 0);
      joined = true;
      executeAfterWait();
      fprintf(stderr, "\t\t\t%s joined\n", getName());
    }
  }
};

#endif // THREADBLE_H
