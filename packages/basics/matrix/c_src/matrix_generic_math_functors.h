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
/// This class is used as FUNCTOR for the templates located at
/// matrix_generic_math_templates.h
template <typename T, typename Func>
class component_wise_functor_0 {
  Func func;
public:
  component_wise_functor_0(Func func) : func(func) { }
  void operator()(Matrix<T> *m,
		  unsigned int size, unsigned int stride,
		  unsigned int offset) const {
    func(size, m->getRawDataAccess(), stride, offset, m->getCudaFlag());
  }
};
/// This make class is needed to do type inference in template classes
template <typename T, typename Func>
component_wise_functor_0<T, Func> make_cwise_functor_0(Func f) {
  return component_wise_functor_0<T, Func>(f);
}
template <typename T, typename Func>
component_wise_functor_0<T, Func>
make_const_cwise_functor_0(Func f) {
  return component_wise_functor_0<const Matrix<T>, Func>(f);
}

///// COMPONENT WISE GENERIC FUNCTOR (ONE ARGUMENTS) /////
/// This class is used as FUNCTOR for the templates located at
/// matrix_generic_math_templates.h
template <typename T, typename Func>
class component_wise_functor_1 {
  T     value;
  Func  func;
public:
  component_wise_functor_1(T value,
			   Func func) : value(value), func(func) { }
  void operator()(Matrix<T> *m,
		  unsigned int size, unsigned int stride,
		  unsigned int offset) const {
    func(size, m->getRawDataAccess(), stride, offset, value, m->getCudaFlag());
  }
};
/// This make class is needed to do type inference in template classes
template <typename T, typename Func>
component_wise_functor_1<T, Func> make_cwise_functor_1(T v, Func f) {
  return component_wise_functor_1<T, Func>(v, f);
}
template <typename T, typename Func>
component_wise_functor_1<T, Func>
make_const_cwise_functor_1(T v, Func f) {
  return component_wise_functor_1<const Matrix<T>, Func>(v, f);
}

#endif // MATRIX_GENERIC_MATH_FUNCTORS_H
