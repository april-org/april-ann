/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef SERIALIZABLE_H
#define SERIALIZABLE_H

#include "generic_options.h"
#include "error_print.h"
#include "referenced.h"
#include "stream.h"
#include "unused_variable.h"

namespace AprilIO {

  /**
   * @brief This class defines the basic API for serializable classes.
   */
  class Serializable : public Referenced {
  public:
    Serializable() : Referenced() { }
    virtual ~Serializable() { }
    
    // The read method needs to be implemented in derived classes.
    // Whatever *read(StreamInterface *dest,
    //                const AprilUtils::GenericOptions *options);

    /// Writes the object data into dest, and it could be retrieved by read.
    virtual void write(StreamInterface *dest,
                       const AprilUtils::GenericOptions *options) = 0;
    
    /**
     * @brief Writes a Lua string which can be instantiated to get the object.
     *
     * Some classes implement this method as derived of this in C++ but others
     * implement it in Lua pure code.
     *
     * @note THIS METHOD IS A FUTURE FEATURE, IT IS NOT BEEN USED ANYWHERE.
     */
    virtual void toLuaString(StreamInterface *dest,
                             const AprilUtils::GenericOptions *options) {
      UNUSED_VARIABLE(dest);
      UNUSED_VARIABLE(options);
      ERROR_EXIT(128, "Unable to instantiate into a Lua string\n");
    }
  };

} // namespace AprilUtils

#endif // SERIALIZABLE_H
