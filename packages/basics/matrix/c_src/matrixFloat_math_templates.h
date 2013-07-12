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
#ifndef MATRIXFLOAT_MATH_TEMPLATES_H
#define MATRIXFLOAT_MATH_TEMPLATES_H

#include <omp.h>
#include "omp_utils.h"
#include "matrix.h"
#include "matrixFloat.h"

// Auxiliary function template which applies a given FUNC object ( implements
// operator() ) to all the elements of a Matrix, using the best_span_iterator,
// and OMP if needed.
template<typename FUNC, typename MATRIX>
void applyFunctionWithSpanIterator(MATRIX *m,
				   const FUNC &functor,
				   const int N_th = 50,
				   const unsigned int SIZE_th = 50) {
  // Contiguous memory block
  if (m->getIsContiguous() && static_cast<unsigned int>(m->size()) < SIZE_th)
    functor(m, static_cast<unsigned int>(m->size()), 1,
	    static_cast<unsigned int>(m->getOffset()));
  // One dimension
  else if (m->getNumDim() == 1)
    functor(m, static_cast<unsigned int>(m->size()),
	    static_cast<unsigned int>(m->getStrideSize(0)),
	    static_cast<unsigned int>(m->getOffset()));
  // General case
  else {
    MatrixFloat::best_span_iterator span_it(m);
    const int N = span_it.numberOfIterations();
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
    if (N > 1) {
      // this if controls the execution using OMP only when the number of threads
      // is more than 1 and the iterator size is big enough
      if (omp_utils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for firstprivate(span_it)
	for (int i=1; i<N; ++i) {
	  span_it.setAtIteration(i);
	  functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
	}
      }
      else {
	// sequential code, with less overhead when updating iterator
	++span_it;
	do {
	  functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
	  ++span_it;
	} while(span_it != m->end_span_iterator());
      }
    }
  }
}

// Idem but for binary functions (needs two matrices, and two
// best_span_iterators)
template<typename FUNC, typename MATRIX1, typename MATRIX2>
void applyBinaryFunctionWithSpanIterator(MATRIX1 *m1,
					 MATRIX2 *m2,
					 const FUNC &functor,
					 const int N_th = 50,
					 const unsigned int SIZE_th = 50) {
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      static_cast<unsigned int>(m1->size()) < SIZE_th)
    functor(m1, m2,
	    static_cast<unsigned int>(m1->size()), 1, 1,
	    static_cast<unsigned int>(m1->getOffset()),
	    static_cast<unsigned int>(m2->getOffset()));
  else if (m1->getNumDim() == 1)
    functor(m1, m2,
	    static_cast<unsigned int>(m1->size()),
	    static_cast<unsigned int>(m1->getStrideSize(0)),
	    static_cast<unsigned int>(m2->getStrideSize(0)),
	    static_cast<unsigned int>(m1->getOffset()),
	    static_cast<unsigned int>(m2->getOffset()));
  else {
    MatrixFloat::best_span_iterator span_it1(m1), span_it2(m2);
    const int N = span_it1.numberOfIterations();
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    assert(N == span_it2.numberOfIterations());
    assert(size == static_cast<unsigned int>(span_it2.getSize()));
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    functor(m1, m2, size, stride1, stride2,
	    static_cast<unsigned int>(span_it1.getOffset()),
	    static_cast<unsigned int>(span_it2.getOffset()));
    if (N > 1) {
      // this if controls the execution using OMP only when the number of threads
      // is more than 1 and the iterator size is big enough
      if (omp_utils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for firstprivate(span_it1) firstprivate(span_it2)
	for (int i=1; i<N; ++i) {
	  span_it1.setAtIteration(i);
	  span_it2.setAtIteration(i);
	  functor(m1, m2, size, stride1, stride2,
		  static_cast<unsigned int>(span_it1.getOffset()),
		  static_cast<unsigned int>(span_it2.getOffset()));
	}
      }
      else {
	// sequential code, with less overhead when updating iterator
	++span_it1;
	++span_it2;
	do {
	  functor(m1, m2, size, stride1, stride2,
		  static_cast<unsigned int>(span_it1.getOffset()),
		  static_cast<unsigned int>(span_it2.getOffset()));
	  ++span_it1;
	  ++span_it2;
	} while(span_it1 != m1->end_span_iterator());
      }
    }
  }
}

// Similar to previous functions, but for sum-reduction operations
template<typename FUNC, typename MATRIX>
float applySumReductionWithSpanIterator(MATRIX *m,
					const FUNC &functor,
					const int N_th = 50,
					const unsigned int SIZE_th = 50) {
  // Contiguous memory block
  if (m->getIsContiguous() && static_cast<unsigned int>(m->size()) < SIZE_th)
    return functor(m, static_cast<unsigned int>(m->size()), 1,
		   static_cast<unsigned int>(m->getOffset()));
  // One dimension
  else if (m->getNumDim() == 1)
    return functor(m, static_cast<unsigned int>(m->size()),
		   static_cast<unsigned int>(m->getStrideSize(0)),
		   static_cast<unsigned int>(m->getOffset()));
  // General case
  else {
    MatrixFloat::best_span_iterator span_it(m);
    const int N = span_it.numberOfIterations();
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    float sum;
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    sum = functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
    if (N > 1) {
      // this if controls the execution using OMP only when the number of threads
      // is more than 1 and the iterator size is big enough
      if (omp_utils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for reduction(+:sum) firstprivate(span_it)
	for (int i=1; i<N; ++i) {
	  span_it.setAtIteration(i);
	  sum += functor(m, size, stride,
			 static_cast<unsigned int>(span_it.getOffset()));
	}
      }
      else {
	// sequential code, with less overhead when updating iterator
	++span_it;
	do {
	  sum += functor(m, size, stride,
			 static_cast<unsigned int>(span_it.getOffset()));
	  ++span_it;
	} while(span_it != m->end_span_iterator());
      }
    }
    return sum;
  }
}

// Similar to previous functions, but for AND-reduction operations (binary)
template<typename FUNC, typename MATRIX1, typename MATRIX2>
bool applyBinaryAndReductionWithSpanIterator(MATRIX1 *m1,
					     MATRIX2 *m2,
					     const FUNC &functor,
					     const int N_th = 50,
					     const unsigned int SIZE_th = 50) {
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      static_cast<unsigned int>(m1->size()) < SIZE_th)
    return functor(m1, m2,
		   static_cast<unsigned int>(m1->size()), 1, 1,
		   static_cast<unsigned int>(m1->getOffset()),
		   static_cast<unsigned int>(m2->getOffset()));
  else if (m1->getNumDim() == 1)
    return functor(m1, m2,
		   static_cast<unsigned int>(m1->size()),
		   static_cast<unsigned int>(m1->getStrideSize(0)),
		   static_cast<unsigned int>(m2->getStrideSize(0)),
		   static_cast<unsigned int>(m1->getOffset()),
		   static_cast<unsigned int>(m2->getOffset()));
  else {
    MatrixFloat::best_span_iterator span_it1(m1), span_it2(m2);
    const int N = span_it1.numberOfIterations();
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    assert(N == span_it2.numberOfIterations());
    assert(size == static_cast<unsigned int>(span_it2.getSize()));
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    bool ret = functor(m1, m2, size, stride1, stride2,
		       static_cast<unsigned int>(span_it1.getOffset()),
		       static_cast<unsigned int>(span_it2.getOffset()));
    if (ret && N > 1) {
      // this if controls the execution using OMP only when the number of threads
      // is more than 1 and the iterator size is big enough
      if (omp_utils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for reduction(&&:ret) firstprivate(span_it1) firstprivate(span_it2)
	for (int i=1; i<N; ++i) {
	  span_it1.setAtIteration(i);
	  span_it2.setAtIteration(i);
	  ret = ret && functor(m1, m2, size, stride1, stride2,
			       static_cast<unsigned int>(span_it1.getOffset()),
			       static_cast<unsigned int>(span_it2.getOffset()));
	}
      }
      else {
	// sequential code, with less overhead when updating iterator
	++span_it1;
	++span_it2;
	do {
	  ret = ret && functor(m1, m2, size, stride1, stride2,
			       static_cast<unsigned int>(span_it1.getOffset()),
			       static_cast<unsigned int>(span_it2.getOffset()));
	  ++span_it1;
	  ++span_it2;
	} while(ret && span_it1 != m1->end_span_iterator());
      }
    }
    return ret;
  }
}

// Similar to previous functions, but for general reduction operations (without
// OMP)
template<typename T, typename MATRIX, typename FUNC1, typename FUNC2>
T applyReductionWithSpanIterator(MATRIX *m,
				 FUNC1 &functor,
				 FUNC2 &reductor,
				 const T initial_value) {
  // Contiguous memory block
  if (m->getIsContiguous())
    return reductor(initial_value,
		    functor(m, static_cast<unsigned int>(m->size()), 1,
			    static_cast<unsigned int>(m->getOffset())));
  // One dimension
  else if (m->getNumDim() == 1)
    return reductor(initial_value,
		    functor(m, static_cast<unsigned int>(m->size()),
			    static_cast<unsigned int>(m->getStrideSize(0)),
			    static_cast<unsigned int>(m->getOffset())));
  // General case
  else {
    MatrixFloat::best_span_iterator span_it(m);
    const int N = span_it.numberOfIterations();
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    T red(initial_value);
    // sequential code
    while(span_it != m->end_span_iterator()) { 
      T result = functor(m, size, stride,
			 static_cast<unsigned int>(span_it.getOffset()));
      red = reductor(red, result);
      ++span_it;
    }
    return red;
  }
}

#endif // MATRIXFLOAT_MATH_TEMPLATES_H
