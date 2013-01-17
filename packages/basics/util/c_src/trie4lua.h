/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef TRIE4LUA_H
#define TRIE4LUA_H

#include "referenced.h"
#include "aux_hash_table.h"
#include "hash_table.h"

namespace april_utils {

  // This c++ class replaces a previous one implemented in lua, I have
  // decided not to reuse a previous trie class since this one is even
  // simpler

  class Trie4lua : public Referenced {
    static const int rootNode = 0;
    int lastId;
    // estado x palabra -> estado
    typedef hash<int_pair, int> hash_type;
    hash_type transition_t;
  public:
    Trie4lua();
    int reserveId();
    int find(int *wordsequence, int lenght);
  };


} // namespace april_utils

#endif // TRIE4LUA_H
