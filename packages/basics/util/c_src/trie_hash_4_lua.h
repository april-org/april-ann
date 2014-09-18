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
#ifndef TRIE_HASH_4_LUA_H
#define TRIE_HASH_4_LUA_H

#include "referenced.h"
#include "aux_hash_table.h"
#include "hash_table.h"

namespace AprilUtils {

  // This c++ class replaces a previous one implemented in lua, I have
  // decided not to reuse a previous trie class since this one is even
  // simpler

  class TrieHash4Lua : public Referenced {
    static const int rootNode = 0;
    int lastId;
    // state x id -> state
    typedef hash<int_pair, int> hash_type;
    hash_type transition_t;
  public:
    TrieHash4Lua();
    int reserveId();
    int find(int *ids, int lenght);
  };


} // namespace AprilUtils

#endif // TRIE_HASH_4_LUA_H
