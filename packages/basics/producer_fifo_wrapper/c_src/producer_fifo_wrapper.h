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
#ifndef PRODUCER_FIFO_WRAPPER_H
#define PRODUCER_FIFO_WRAPPER_H

#include "function_interface.h"
#include "mutexed_fifo.h"

using AprilThreadUtils::MutexedFIFO;
using namespace Functions;

class ProducerFifoWrapper : public DataProducer<double> {
  MutexedFIFO<double*> queue;
public:
  ProducerFifoWrapper();
  ~ProducerFifoWrapper();

  // metodos para dar cuenta de dataproducer
  double *get();
  void reset();

  // metodos wrapper de mutexedfifo
  void unlock();
  void lock();

  // metodo para bindear a lua:
  void put(double*);
};

#endif // PRODUCER_FIFO_WRAPPER_H

