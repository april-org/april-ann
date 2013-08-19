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
#ifndef MATRIX_GENERIC_MATH_FUNCTORS_H
#define MATRIX_GENERIC_MATH_FUNCTORS_H

#include "matrix.h"

///// COMPONENT WISE GENERIC FUNCTOR (NO ARGUMENTS) /////
/// This macro defines structs to be used as FUNCTOR for the templates located
/// at matrix_generic_math_templates.h
#define DEF_CWISE_FUNCTOR_0(FUNC,TYPE)				\
  struct FUNC##TYPE##CWiseFunctor0 {				\
    FUNC##TYPE##CWiseFunctor0() { }				\
    ~FUNC##TYPE##CWiseFunctor0() { }				\
    void operator()(Matrix<TYPE> *m,				\
		    unsigned int size, unsigned int stride,	\
		    unsigned int offset) const {		\
      FUNC(size, m->getRawDataAccess(), stride, offset,		\
	   m->getCudaFlag());					\
    }								\
  }
/// This instantiates a variable of the previous struct
#define MAKE_CWISE_FUNCTOR_0(FUNC,TYPE) FUNC##TYPE##CWiseFunctor0()


///// COMPONENT WISE GENERIC FUNCTOR (ONE ARGUMENT) /////
/// This macro defines structs to be used as FUNCTOR for the templates located
/// at matrix_generic_math_templates.h
#define DEF_CWISE_FUNCTOR_1(FUNC,TYPE)				\
  struct FUNC##TYPE##CWiseFunctor1 {				\
    TYPE value;							\
    FUNC##TYPE##CWiseFunctor1(TYPE v) : value(v) { }		\
    ~FUNC##TYPE##CWiseFunctor1() { }				\
    void operator()(Matrix<TYPE> *m,				\
		    unsigned int size, unsigned int stride,	\
		    unsigned int offset) const {		\
      FUNC(size, m->getRawDataAccess(), stride, offset,		\
	   value, m->getCudaFlag());				\
    }								\
  }
/// This instantiates a variable of the previous struct
#define MAKE_CWISE_FUNCTOR_1(FUNC,TYPE,VALUE) FUNC##TYPE##CWiseFunctor1(VALUE)

#endif // MATRIX_GENERIC_MATH_FUNCTORS_H
