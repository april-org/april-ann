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
#include "luabindutil.h"
#include "luabindmacros.h"

// This function is to be used with LUABIND_GET_.... macros
bool lua_isComplexF(lua_State *L, int n) {
  return lua_isLuaComplexFNumber(L,n) || lua_isstring(L,n);
}

// This function is to be used with LUABIND_GET_.... macros
#define FUNCTION_NAME "lua_toComplexF"
ComplexF lua_toComplexF(lua_State *L, int n) {
  if (lua_isLuaComplexFNumber(L, n)) {
    LuaComplexFNumber *aux;
    LUABIND_GET_PARAMETER(n, LuaComplexFNumber, aux);
    return aux->number;
  }
  else if (lua_isstring(L, n)) {
    const char *str;
    LUABIND_GET_PARAMETER(n, string, str);
    LuaComplexFNumber aux(str);
    return aux.number;
  }
  else LUABIND_ERROR("Incorrect Lua complex representation, use string "
		     "or complex object");
  return ComplexF();
}
#undef FUNCTION_NAME

void lua_pushComplexF(lua_State *L, const ComplexF &number) {
  LuaComplexFNumber *obj = new LuaComplexFNumber(number);
  lua_pushLuaComplexFNumber(L, obj);
}
//BIND_END

//BIND_HEADER_H
#include "referenced.h"
#include "complex_number.h"
#include "error_print.h"
#include "constString.h"

struct LuaComplexFNumber : public Referenced {
  ComplexF number;
  LuaComplexFNumber(const ComplexF &number) : Referenced(), number(number) { }
  LuaComplexFNumber(const char *str) : Referenced() {
    constString cs(str);
    float a, b;
    char i;
    if (!cs.extract_float(&a)) {
      char sgn;
      // not number: only allowed i, +i or -i
      number.real() = 0.0f;
      if (!cs.extract_char(&sgn))
	ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
      // not signed
      if (sgn == 'i') number.img() = 1.0f;
      else {
	// signed
	if (sgn!='-' && sgn!='+')
	  ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
	if (!cs.extract_char(&i))
	  ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
	if (i != 'i')
	  ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
	if (sgn == '-') number.img() = -1.0f;
	else number.img() = 1.0f;
      }
    }
    else if (!cs.extract_float(&b)) {
      // only one number
      if (!cs.extract_char(&i)) {
	// without i char
	number.real() = a;
	number.img()  = 0.0f;
      }
      else {
	if (i != 'i')
	  ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
	// with i char
	number.real() = 0.0f;
	number.img()  = a;
      }
    }
    else {
      // two numbers
      if (!cs.extract_char(&i))
	ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
      if (i != 'i')
	ERROR_EXIT1(256, "Incorrect complex string format '%s'\n",str);
      number.real() = a;
      number.img()  = b;
    }
  }
};

// This function is to be used with LUABIND_GET_.... macros
ComplexF lua_toComplexF(lua_State *L, int n);
// This function is to be used with LUABIND_GET_.... macros
bool lua_isComplexF(lua_State *L, int n);
void lua_pushComplexF(lua_State *L, const ComplexF &number);
//BIND_END

//BIND_LUACLASSNAME LuaComplexFNumber complex
//BIND_CPP_CLASS    LuaComplexFNumber

//BIND_CONSTRUCTOR LuaComplexFNumber
{
  int argn = lua_gettop(L); // number of arguments
  if (argn == 2) {
    ComplexF number;
    LUABIND_GET_PARAMETER(1, float, number.real());
    LUABIND_GET_PARAMETER(2, float, number.img());
    obj = new LuaComplexFNumber(number);
  }
  else if (argn == 1) {
    const char *str;
    LUABIND_GET_PARAMETER(1, string, str);
    obj = new LuaComplexFNumber(str);
  }
  else {
    LUABIND_FERROR1("Incorrect number of arguments, expected 1 or 2, found %d",
		    argn);
  }
  LUABIND_RETURN(LuaComplexFNumber, obj);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber clone
{
  LuaComplexFNumber *other = new LuaComplexFNumber(obj->number);
  LUABIND_RETURN(LuaComplexFNumber, other);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber eq
{
  LuaComplexFNumber *other;
  LUABIND_GET_PARAMETER(1, LuaComplexFNumber, other);
  LUABIND_RETURN(bool, obj->number == other->number);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber mul
{
  LuaComplexFNumber *other, *result;
  LUABIND_GET_PARAMETER(1, LuaComplexFNumber, other);
  result = new LuaComplexFNumber( obj->number * other->number );
  LUABIND_RETURN(LuaComplexFNumber, result);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber add
{
  LuaComplexFNumber *other, *result;
  LUABIND_GET_PARAMETER(1, LuaComplexFNumber, other);
  result = new LuaComplexFNumber( obj->number + other->number );
  LUABIND_RETURN(LuaComplexFNumber, result);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber sub
{
  LuaComplexFNumber *other, *result;
  LUABIND_GET_PARAMETER(1, LuaComplexFNumber, other);
  result = new LuaComplexFNumber( obj->number - other->number );
  LUABIND_RETURN(LuaComplexFNumber, result);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber div
{
  LuaComplexFNumber *other, *result;
  LUABIND_GET_PARAMETER(1, LuaComplexFNumber, other);
  result = new LuaComplexFNumber( obj->number / other->number );
  LUABIND_RETURN(LuaComplexFNumber, result);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber neg
{
  LuaComplexFNumber *result;
  result = new LuaComplexFNumber( -obj->number );
  LUABIND_RETURN(LuaComplexFNumber, result);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber conj
{
  obj->number.conj();
  LUABIND_RETURN(LuaComplexFNumber, obj);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber exp
{
  LuaComplexFNumber *result = new LuaComplexFNumber( obj->number.expc() );
  LUABIND_RETURN(LuaComplexFNumber, result);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber real
{
  LUABIND_RETURN(float, obj->number.real());
}
//BIND_END

//BIND_METHOD LuaComplexFNumber img
{
  LUABIND_RETURN(float, obj->number.img());
}
//BIND_END

//BIND_METHOD LuaComplexFNumber plane
{
  LUABIND_RETURN(float, obj->number.real());
  LUABIND_RETURN(float, obj->number.img());
}
//BIND_END

//BIND_METHOD LuaComplexFNumber polar
{
  float r, phi;
  obj->number.polar(r, phi);
  LUABIND_RETURN(float, r);
  LUABIND_RETURN(float, phi);
}
//BIND_END

//BIND_METHOD LuaComplexFNumber abs
{
  LUABIND_RETURN(float, obj->number.abs());
}
//BIND_END

//BIND_METHOD LuaComplexFNumber sqrt
{
  LUABIND_RETURN(float, obj->number.sqrtc());
}
//BIND_END
