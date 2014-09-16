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
//BIND_HEADER_H
#include "threadable.h"
#include "sig_int_handler.h"
using namespace AprilThreadUtils;
//BIND_END

//BIND_HEADER_C
//BIND_END

//BIND_FUNCTION util.threads.install_sig_int_handler
{
  SigIntHandler::install_handler();
}
//BIND_END

//BIND_LUACLASSNAME Threadable __threadable__
//BIND_CPP_CLASS    Threadable

//BIND_CONSTRUCTOR Threadable
{
  LUABIND_ERROR("Abstract class!!");
}
//BIND_END

//BIND_METHOD Threadable is_running
{
  LUABIND_RETURN(bool, obj->isRunning());
}
//BIND_END

//BIND_METHOD Threadable is_joined
{
  LUABIND_RETURN(bool, obj->isJoined());
}
//BIND_END

//BIND_METHOD Threadable start_thread
{
  obj->startThread();
}
//BIND_END

//BIND_METHOD Threadable stop_thread
{
  obj->stopThread();
}
//BIND_END

//BIND_METHOD Threadable wait_thread
{
  obj->waitThread();
}
//BIND_END

//BIND_METHOD Threadable get_name
{
  LUABIND_RETURN(string, obj->getName());
}
//BIND_END
