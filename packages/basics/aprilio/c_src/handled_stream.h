/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef HANDLED_STREAM_H
#define HANDLED_STREAM_H

#include "stream.h"

namespace AprilIO {
  
  /**
   * @brief Introduces abstract methods useful for streams based on file handler
   * descriptor.
   *
   * @note It is not derived from StreamInterface so any derived class must be
   * also derived from any kind of StreamInterface classes.
   */
  class HandledStreamInterface {
  public:
    HandledStreamInterface() { }
    virtual ~HandledStreamInterface() { }
    /// Returns the underlying file handler descriptor.
    virtual int fileno() const = 0;
  };
  
} // namespace AprilIO

#endif // HANDLED_STREAM_H
