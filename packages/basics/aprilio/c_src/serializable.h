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

extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

#include "error_print.h"
#include "lua_table.h"
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
    //                const AprilUtils::LuaTable &options);

    /// Writes the object data into dest, and it could be retrieved by read.
    virtual void write(StreamInterface *dest,
                       const AprilUtils::LuaTable &options) {
      UNUSED_VARIABLE(dest);
      UNUSED_VARIABLE(options);
      ERROR_EXIT(128, "Unable to use write method\n");
    }

    /**
     * @brief Returns the class constructor for serialized objects.
     *
     * This method is needed in C++ to allow polymorphic calls from Lua side.
     */
    virtual const char *luaCtorName() const {
      ERROR_EXIT(128, "Unable to retrieve Lua ctor name\n");
      return 0;
    }

    /**
     * @brief Stores into an AprilUtils::LuaTable all the parameters of this
     * object.
     *
     * The parameters will use to serialize the object, and them will be given
     * to the Lua binding constructor to retrieve a serialized object. This
     * method is needed in C++ to allow polymorphic calls from Lua side.
     *
     * @returns The number of items pushed into Lua stack.
     */
    virtual int exportParamsToLua(lua_State *L) {
      UNUSED_VARIABLE(L);
      return 0;
    }
  };

} // namespace AprilUtils

#endif // SERIALIZABLE_H
