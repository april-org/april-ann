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
//BIND_HEADER_C
#include "bind_function_interface.h"
//BIND_END

//BIND_HEADER_H
#include "producer_fifo_wrapper.h"
//BIND_END

//BIND_LUACLASSNAME ProducerFifoWrapper utils.producer_fifo_wrapper
//BIND_CPP_CLASS    ProducerFifoWrapper

//BIND_LUACLASSNAME DoubleDataProducer   __double_data_producer__
//BIND_SUBCLASS_OF  ProducerFifoWrapper DoubleDataProducer

//BIND_CONSTRUCTOR  ProducerFifoWrapper
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(ProducerFifoWrapper, new ProducerFifoWrapper());
}
//BIND_END

//BIND_METHOD ProducerFifoWrapper put
{
  LUABIND_CHECK_ARGN(==, 1);
  int    vec_size = lua_objlen(L, 1);
  double *vec      = new double[vec_size];
  LUABIND_TABLE_TO_VECTOR(1, double, vec, vec_size);
  obj->put(vec);
}
//BIND_END

//BIND_METHOD ProducerFifoWrapper lock
{
  LUABIND_CHECK_ARGN(==, 0);
  obj->lock();
}
//BIND_END

//BIND_METHOD ProducerFifoWrapper unlock
{
  LUABIND_CHECK_ARGN(==, 0);
  obj->unlock();
}
//BIND_END

