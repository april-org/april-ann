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
#ifndef CONSTSTRING_H
#define CONSTSTRING_H

#include <cstdio>
#include <stdint.h>
#include "logbase.h"

class constString {
	const char *buffer;
	size_t length;
 public:
	constString() { buffer = 0; length = 0; }
	constString(const char *s);
	constString(const char *s, size_t n);
	operator bool()   { return  length>0 &&  buffer; }
	bool empty() const{ return length<=0 || !buffer; }
	bool operator !() { return length<=0 || !buffer; }
	size_t len() const { return length; }
	// ojito, este método NO añade el \0 al final
	operator const char *() const { return buffer; }

	char operator [] (int i) const { return buffer[i]; }
	char operator [] (unsigned int i) const { return buffer[i]; }
	// char operator [] (size_t i) const { return buffer[i]; }

	char *newString() const;

	// el método siguiente crea una cadena nueva que se debe
	// liberar posteriormente con delete[]

	// por implementar:
	// char* operator char*() { return buffer; } // conversor
	bool operator == (const constString &otro) const;
	bool operator != (const constString &otro) const;
	bool operator <  (const constString &otro) const;
	bool operator <= (const constString &otro) const;
	bool operator >  (const constString &otro) const;
	bool operator >= (const constString &otro) const;
	bool operator == (const char *otro) const { return operator==(constString(otro)); }
	bool operator != (const char *otro) const { return operator!=(constString(otro)); }
	bool operator <  (const char *otro) const { return operator<(constString(otro)); }
	bool operator <= (const char *otro) const { return operator<=(constString(otro)); }
	bool operator >  (const char *otro) const { return operator>(constString(otro)); }
	bool operator >= (const char *otro) const { return operator>=(constString(otro)); }
 	void skip(size_t n);
	void ltrim(const char *acepta = " \t");
// 	void rtrim();
// 	void trim();
	bool is_prefix(const constString &theprefix);
	bool is_prefix(const char *theprefix);
	bool skip(const char *theprefix);
	bool skip(const constString &theprefix);
	constString extract_prefix(size_t prefix_len);
	constString extract_token(const char *separadores=" \t,;\r\n");
	constString extract_line();
	constString extract_u_line();
	bool extract_char(char *resul);
	bool extract_int(int *resul, int base = 10, 
			 const char *separadores=" \t,;\r\n");
	bool extract_unsigned_int(unsigned int *resul, int base = 10, 
				  const char *separadores=" \t,;\r\n");
	bool extract_long(long int *resul, int base = 10, 
			  const char *separadores=" \t,;\r\n");
	bool extract_long_long(long long int *resul, int base = 10, 
			       const char *separadores=" \t,;\r\n");
	bool extract_float(float *resul, 
			   const char *separadores=" \t,;\r\n");
	bool extract_double(double *resul, 
			    const char *separadores=" \t,;\r\n");
	bool extract_float_binary(float *resul);
	bool extract_double_binary(double *resul);

	bool extract_log_float_binary(log_float *resul);
	bool extract_log_double_binary(log_double *resul);
#ifdef HAVE_UINT16
	bool extract_uint16_binary(uint16_t *resul);
	bool extract_int16_binary(int16_t *resul);
#endif
	bool extract_uint32_binary(uint32_t *resul);
	bool extract_int32_binary(int32_t *resul);
#ifdef HAVE_UINT64
	bool extract_uint64_binary(uint64_t *resul);
	bool extract_int64_binary(int64_t *resul);
#endif
	void print(FILE *F=stdout);
        bool good() const { return !empty(); }
};

char *copystr(const char *c);

#endif // CONSTSTRING_H
