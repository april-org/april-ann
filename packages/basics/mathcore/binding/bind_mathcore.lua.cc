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
//BIND_HEADER_C
#include "gpu_mirrored_memory_block.h"
//BIND_END

//BIND_HEADER_H
//BIND_END

//BIND_FUNCTION mathcore.set_mmap_allocation
{
  bool v;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, bool, v);
  GPUMirroredMemoryBlockBase::setUseMMapAllocation(v);
}
//BIND_END

//BIND_FUNCTION mathcore.set_max_pool_size
{
  int max_pool_size;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, max_pool_size);
#ifndef NO_POOL
  GPUMirroredMemoryBlockBase::
    changeMaxPoolSize(static_cast<size_t>(max_pool_size));
#endif
}
//BIND_END
