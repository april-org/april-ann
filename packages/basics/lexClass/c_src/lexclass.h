/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Francisco Zamora-Martinez, Salvador España-Boquera
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
#ifndef LEXCLASS_H
#define LEXCLASS_H

#include <stdint.h>
#include "vector.h"
#include "referenced.h"
#include "hash_table.h"
#include "aux_hash_table.h"

namespace AprilUtils {

  // OJO! esta tabla empieza en 1. el 0 esta reservado
  struct LexClass : public Referenced {
    
    struct data_t {
      uint32_t   word, outsym;
      log_float  prob;
    };

    vector<data_t>		data;
    vector<char *>		strings;
    vector<int_pair>		string2wid_oid;
    vector<uint32_t>		wid2string;
    vector<uint32_t>		oid2string;
    hash<char *,uint32_t>	string2id;
    
    LexClass() {
      char *aux1 = new char[2];
      aux1[0] = '_'; aux1[1] = '\0';
      char *aux2 = new char[2];
      aux2[0] = '_'; aux2[1] = '\0';
      // el 0 esta reservado para palabra vacia
      addPair(aux1, aux2, log_float::one());
    }
    ~LexClass() {
      for (unsigned int i=0; i<strings.size(); ++i)
	delete[] strings[i];
    }

    uint32_t addPair(char *word,
		     char *outsym,
		     log_float prob) {
      uint32_t wid, oid;
      if (word != 0 && strlen(word) > 0) {
	uint32_t &string_id = string2id[word];
	if (string_id == 0) {
	  // no estaba en la tabla
	  string_id = strings.size();
	  strings.push_back(word);
	  string2wid_oid.push_back(int_pair(0,0));
	}
	else delete[] word;
	if ((wid=string2wid_oid[string_id].first) == 0) {
	  string2wid_oid[string_id].first = wid2string.size();
	  wid				  = wid2string.size();
	  wid2string.push_back(string_id);
	}
      }
      else wid = 0;
      if (outsym != 0 && strlen(outsym) > 0) {
	uint32_t &string_id = string2id[outsym];
	if (string_id == 0) {
	  // no estaba en la tabla
	  string_id = strings.size();
	  strings.push_back(outsym);
	  string2wid_oid.push_back(int_pair(0,0));
	}
	else delete[] outsym;
	if ((oid=string2wid_oid[string_id].second) == 0) {
	  string2wid_oid[string_id].second = oid2string.size();
	  oid				   = oid2string.size();
	  oid2string.push_back(string_id);
	}
      }
      else oid = 0;
      
      data_t	d;
      d.word   = wid;
      d.outsym = oid;
      d.prob   = prob;
      data.push_back(d);
      //printf("%u: %s %s (%u %u)\n",
      //data.size()-1, strings[wid2string[wid]], strings[oid2string[oid]],
      //wid, oid);
      return data.size()-1;
    }
    unsigned int size() { return data.size() - 1; }

    const char *getWordFromWordId(uint32_t wid) {
      return strings[ wid2string[wid]];
    }
    
    const char *getWordFromPairId(uint32_t pairid) {
      return getWordFromWordId(data[pairid].word);
    }
    
    const char *getOutSymFromOutSymId(uint32_t oid) {
      return strings[ oid2string[oid]];
    }

    const char *getOutSymFromPairId(uint32_t pairid) {
      return getOutSymFromOutSymId(data[pairid].outsym);
    }
    
    bool getWordId(const char *word, uint32_t &wid) {
      uint32_t *string_id;
      string_id = string2id.find((char*)word);
      if (string_id == 0) return false;
      wid = string2wid_oid[*string_id].first;
      return true;
    }

    bool getOutSymId(const char *outsym, uint32_t &oid) {
      uint32_t *string_id;
      string_id = string2id.find((char*)outsym);
      if (string_id == 0) return false;
      oid = string2wid_oid[*string_id].second;
      return true;
    }
    
    data_t getPairData(uint32_t pair_id) {
      return data[pair_id];
    }
    
    unsigned int wordTblSize() {
      return wid2string.size()-1;
    }

    unsigned int outsymTblSize() {
      return oid2string.size()-1;
    }
    
  };
  
}

#endif
