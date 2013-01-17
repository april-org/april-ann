/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef MUTEX_FIFO_H
#define MUTEX_FIFO_h

#include <pthread.h>
#include <unistd.h>
#include "fifo.h"

using april_utils::fifo;

namespace april_thread_utils {


  template<typename T>
  class MutexedFIFO {
    // cola
    fifo<T>         queue;
    // semaforo
    pthread_mutex_t fifo_mutex;
    // variable condicional para que se esperen los procesos lectores a
    // que haya datos en el canal.
    pthread_cond_t  fifo_readers_cond;
    bool            locked;

  public:
    MutexedFIFO() {
      locked    = true;
      pthread_mutex_init(&fifo_mutex,        NULL);
      pthread_cond_init (&fifo_readers_cond, NULL);
    }
    ~MutexedFIFO() {
      pthread_mutex_destroy(&fifo_mutex);
      pthread_cond_destroy(&fifo_readers_cond);
    }
  
    // indica si esta vacia la cola
    bool empty() {
      bool ret;
      pthread_mutex_lock(&fifo_mutex);
      ret = queue.empty();
      pthread_mutex_unlock(&fifo_mutex);
      return ret;
    }
  
    // ESTE METODO SI ES BLOQUEANTE EN FUNCION DEL VALOR DE
    // "locked". Si es true, entonces es bloqueante, en otro caso no
    T get() {
      pthread_mutex_lock(&fifo_mutex);
      // metemos la espera en un bucle para comprobar cada vez que se
      // despierte si realmente esta vacia la cola
      while(queue.empty()) {
	if (!locked) {
	  // salimos sin bloquear
	  pthread_mutex_unlock(&fifo_mutex);
	  return T();
	}
	// bloqueamos
	pthread_cond_wait(&fifo_readers_cond, &fifo_mutex);
      }
      T data = T();
      queue.get(data);
      pthread_mutex_unlock(&fifo_mutex);
      return data;
    }
  
    // hace un put en el fifo, usando el mutex y despertando a quien
    // toque si hubiera alguien esperando. Este nunca es bloqueante.
    void put(T data) {
      pthread_mutex_lock(&fifo_mutex);
      queue.put(data);
      if (queue.size() == 1) {
	// SUPERFIXME se puede poner el pthread_cond_signal a pelo
	// enviamos la senyal de que se puede leer del canal
	pthread_cond_signal(&fifo_readers_cond);
      }
      pthread_mutex_unlock(&fifo_mutex);
    }
    
    // desbloquea el fifo despertando los procesos dormidos,
    // impidiendo que vuelva a ser bloqueante
    void unlock() {
      locked = false;
      pthread_cond_signal(&fifo_readers_cond);
    }
    
    // bloquea el fifo. vuelve a ser bloqueante
    void lock() {
      locked = true;
    }

  };

};

#endif // MUTEX_FIFO_H
