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
#include "trie_hash_4_lua.h"

namespace AprilUtils {

  TrieHash4Lua::TrieHash4Lua() {
    lastId = rootNode;
  }
   
  int TrieHash4Lua::reserveId() {
    lastId++;
    return lastId;
  }

  int TrieHash4Lua::find(int *ids, int lenght) {
    int index = rootNode;
    for (int i=0; i<lenght; ++i) {
      bool isNew;
      hash_type::value_type *r;
      r = transition_t.find_and_add_pair(make_pair(index,ids[i]),isNew);
      if (isNew) {
	r->second = reserveId();
      }
      index = r->second;
    }
    return index;
  }

} // namespace

