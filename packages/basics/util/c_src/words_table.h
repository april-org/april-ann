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
#ifndef WORDS_TABLE_H
#define WORDS_TABLE_H

#include <cstdio> // para debug
#include "referenced.h"
#include "vector.h"

using AprilUtils::vector;

namespace AprilUtils {
  class WordsTable : public Referenced {
    vector<unsigned int> table_first_index;
    vector<unsigned int> words;
    
  public:
    WordsTable() {
      /** la palabra 0 esta reservada **/
      table_first_index.push_back(0);
      /** centinela **/
      table_first_index.push_back(0);
    }
  
    WordsTable(WordsTable *other, unsigned int *filter) {
      for (unsigned int i=0; i<other->table_first_index.size(); ++i) {
	table_first_index.push_back(other->table_first_index[i]);
      }
      words.reserve(other->words.size());
      for (unsigned int i=0; i<other->words.size(); ++i) {
	words[i] = filter[other->words[i]];
      }
    }
  
    ~WordsTable() {
    }
    
    void insertWords(unsigned int *vec, unsigned int size) {
      for (unsigned int i=0; i<size; ++i) {
	words.push_back(vec[i]);
      }
      /** centinela **/
      table_first_index.push_back(words.size());
    }
    
    unsigned int getWords(unsigned int index, unsigned int **vec) {
      *vec = &words[table_first_index[index]];
      return table_first_index[index+1] - table_first_index[index];
    }
    
    unsigned int size() const {
      return table_first_index.size() - 2;
    }
  };
}

#endif // WORDS_TABLE_H
