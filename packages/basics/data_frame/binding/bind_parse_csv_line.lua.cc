/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
extern "C" {
#include <ctype.h>
}
#include "smart_ptr.h"
#include "constString.h"
//BIND_END
//BIND_HEADER_C
using namespace AprilUtils;

bool isnumber(constString tk, const char *decimal) {
  if (tk[0] == '+' || tk[0] == '-') tk.skip(1);
  if (!tk) return false;
  while(isdigit(tk[0])) tk.skip(1);
  if (!tk) return true;
  if (tk[0] == *decimal) tk.skip(1);
  if (!tk) return true;
  while(isdigit(tk[0])) tk.skip(1);
  if (!tk) return true;
  if (tk[0] == 'e' || tk[0] == 'E') tk.skip(1);
  if (tk[0] == '+' || tk[0] == '-') tk.skip(1);
  if (!tk) return false;
  while(isdigit(tk[0])) tk.skip(1);
  if (!tk) return true;
  return false;
}

double tonumber(constString tk, const char *decimal) {
  char *aux;
  double result;
  if (*decimal != '.') {
    AprilUtils::UniquePtr<char []> new_tk = tk.newString();
    for (size_t i=0; i<tk.len(); ++i) {
      if (new_tk[i] == *decimal) {
        new_tk[i] = '.';
        break;
      }
    }
    result = strtod(new_tk.get(), &aux);
  }
  else {
    result = strtod((const char *)tk, &aux);
  }
  return result;
}
//BIND_END

//BIND_FUNCTION util.__parse_csv_line__
{
  const int t_pos=1; // a destination table is at stack position 1
  constString line = lua_toconstString(L,1+t_pos); // line needs to end with sep
  const char *sep = lua_tostring(L,2+t_pos);
  const char *quotechar = lua_tostring(L,3+t_pos);
  const char *decimal = lua_tostring(L,4+t_pos);
  constString NA_str = lua_toconstString(L,5+t_pos);
  const int NA_pos=6+t_pos; // NA is at stack position 7
  constString tk;
  int n = 0;
  do {
    ++n;
    // next field is quoted? (start with quotechar?) 
    if (line[0] == *quotechar) {
      line.skip(1);
      tk = line.extract_token(quotechar, true);
      if (tk[tk.len()-1] != *quotechar) {
        LUABIND_FERROR1("unmatched %c", *quotechar);
      }
      line.ltrim(sep);
    }
    else { // unquoted; find next sep
      tk = line.extract_token(sep, true);
    }
    tk = tk.extract_prefix(tk.len()-1);
    if (!tk || tk == NA_str) { // NA field, replaced by nan
      lua_pushvalue(L, NA_pos);
    }
    else if (isnumber(tk, decimal)) {
      lua_pushnumber(L, tonumber(tk, decimal));
    }
    else {
      lua_pushlstring(L, (const char *)tk, tk.len());
    }
    lua_rawseti(L, t_pos, n);
  } while (line);
  if (static_cast<size_t>(n) != lua_rawlen(L, t_pos)) {
    LUABIND_ERROR("Incorrect number of columns");
  }
  lua_pushvalue(L, t_pos);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

