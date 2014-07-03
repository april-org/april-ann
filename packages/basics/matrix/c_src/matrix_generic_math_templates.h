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
#ifndef MATRIX_GENERIC_MATH_TEMPLATES_H
#define MATRIX_GENERIC_MATH_TEMPLATES_H

#include "unused_variable.h"
#include "omp_utils.h"
#include "matrix.h"

#define DEFAULT_N_TH 100
#define DEFAULT_SIZE_TH 100

// Auxiliary function template which applies a FUNCTOR REDUCTION over a given
// matrix, so the functor is called as FUNC(a matrix), and returns a type O,
// the type of the destination matrix
template<typename T, typename O, typename FUNC>
Matrix<O> *applyFunctorOverDimension(FUNC func,
				     Matrix<T> *orig,
				     int dim,
				     Matrix<O> *dest=0) {
  const int numDim      = orig->getNumDim();
  const int *matrixSize = orig->getDimPtr();
  int *result_dims      = new int[numDim];
  /**** ORIG sliding window ****/
  int *orig_w_size      = new int[numDim];
  int *orig_w_num_steps = new int[numDim];
  int result_size=1;
  for (int i=0; i<dim; ++i) {
    orig_w_size[i] = 1;
    result_dims[i] = orig_w_num_steps[i] = matrixSize[i];
    result_size *= result_dims[i];
  }
  result_dims[dim] = 1;
  orig_w_size[dim] = matrixSize[dim];
  orig_w_num_steps[dim] = 1;
  for (int i=dim+1; i<numDim; ++i) {
    orig_w_size[i] = 1;
    result_dims[i] = orig_w_num_steps[i] = matrixSize[i];
  }
  typename Matrix<T>::sliding_window orig_w(orig,orig_w_size,0,0,orig_w_num_steps,0);
  Matrix<T> *slice = orig_w.getMatrix();
  IncRef(slice);
  /******************************/
  Matrix<O> *result = dest;
  if (result == 0) result = new Matrix<O>(numDim, result_dims,
					  orig->getMajorOrder());
  else if (result->size() != result_size)
    // else if (!result->sameDim(result_dims, numDim))
    ERROR_EXIT2(256, "Incorrect size at the given dest matrtix, "
		"expected %d, found %d\n", result_size, result->size());
  // traverse in row major order
  for (typename Matrix<O>::iterator it(result->begin());
       it!=result->end(); ++it) {
    orig_w.getMatrix(slice);
    *it = func(slice);
    orig_w.next();
  }
  DecRef(slice);
  delete[] orig_w_size;
  delete[] orig_w_num_steps;
  delete[] result_dims;
  return result;
}

// Auxiliary function template which applies a FUNCTOR REDUCTION over a given
// matrix, so the functor is called as FUNC(a matrix, a matrix component), and
// returns a type O, the type of the destination matrix
template<typename T, typename O, typename C, typename FUNC>
Matrix<O> *applyFunctorOverDimension2(FUNC func,
				      Matrix<T> *orig,
				      int dim,
				      Matrix<O> *dest,
				      Matrix<C> *other) {
  if (other == 0)
    ERROR_EXIT(256, "Other dest matrix not given, use a different template\n");
  const int numDim      = orig->getNumDim();
  const int *matrixSize = orig->getDimPtr();
  int *result_dims      = new int[numDim];
  /**** ORIG sliding window ****/
  int *orig_w_size      = new int[numDim];
  int *orig_w_num_steps = new int[numDim];
  int result_size=1;
  for (int i=0; i<dim; ++i) {
    orig_w_size[i] = 1;
    result_dims[i] = orig_w_num_steps[i] = matrixSize[i];
    result_size *= result_dims[i];
  }
  result_dims[dim] = 1;
  orig_w_size[dim] = matrixSize[dim];
  orig_w_num_steps[dim] = 1;
  for (int i=dim+1; i<numDim; ++i) {
    orig_w_size[i] = 1;
    result_dims[i] = orig_w_num_steps[i] = matrixSize[i];
  }
  typename Matrix<T>::sliding_window orig_w(orig,orig_w_size,0,0,orig_w_num_steps,0);
  Matrix<T> *slice = orig_w.getMatrix();
  IncRef(slice);
  /******************************/
  Matrix<O> *result = dest;
  if (result == 0) result = new Matrix<O>(numDim, result_dims,
					  orig->getMajorOrder());
  else if (result->size() != result_size)
    // else if (!result->sameDim(result_dims, numDim))
    ERROR_EXIT2(256, "Incorrect size at the given dest matrtix, "
		"expected %d, found %d\n", result_size, result->size());
  if (result->size() != other->size())
    ERROR_EXIT(256, "Incorrect size at the given other dest matrtix\n");
  // traverse in row major order
  typename Matrix<C>::iterator other_it(other->begin());
  for (typename Matrix<O>::iterator it(result->begin());
       it!=result->end(); ++it, ++other_it) {
    orig_w.getMatrix(slice);
    *it = func(slice, *other_it);
    orig_w.next();
  }
  DecRef(slice);
  delete[] orig_w_size;
  delete[] orig_w_num_steps;
  delete[] result_dims;
  return result;
}

// Auxiliary function template which applies a given FUNC object ( implements
// operator() ) to all the elements of a Matrix, using the best_span_iterator,
// and OMP if needed.
template<typename T, typename FUNC>
void applyFunctionWithSpanIterator(Matrix<T> *m,
				   const FUNC &functor,
				   const int N_th = DEFAULT_N_TH,
				   const unsigned int SIZE_th = DEFAULT_SIZE_TH) {
#ifdef NO_OMP
  UNUSED_VARIABLE(N_th);
  UNUSED_VARIABLE(SIZE_th);
#endif
  // Contiguous memory block
  if (m->getIsContiguous())
    functor(m, static_cast<unsigned int>(m->size()), 1,
	    static_cast<unsigned int>(m->getOffset()));
  // One dimension
  else if (m->getNumDim() == 1)
    functor(m, static_cast<unsigned int>(m->size()),
	    static_cast<unsigned int>(m->getStrideSize(0)),
	    static_cast<unsigned int>(m->getOffset()));
  // General case
  else {
    typename Matrix<T>::best_span_iterator span_it(m);
    const int N = span_it.numberOfIterations();
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
    if (N > 1) {
#ifndef NO_OMP
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
#endif
	// sequential code, with less overhead when updating iterator
	++span_it;
	do {
	  functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
	  ++span_it;
	} while(span_it != m->end_span_iterator());
#ifndef NO_OMP
      }
#endif
    }
  }
}

// Idem but for binary functions (needs two matrices, and two
// best_span_iterators)
template<typename T, typename FUNC>
void applyBinaryFunctionWithSpanIterator(Matrix<T> *m1,
					 Matrix<T> *m2,
					 const FUNC &functor,
					 const int N_th = DEFAULT_N_TH,
					 const unsigned int SIZE_th = DEFAULT_SIZE_TH) {
#ifdef NO_OMP
  UNUSED_VARIABLE(N_th);
  UNUSED_VARIABLE(SIZE_th);
#endif
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      m1->getIsDataRowOrdered() == m2->getIsDataRowOrdered())
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
    typename Matrix<T>::best_span_iterator span_it1(m1), span_it2(m2);
    const int N = span_it1.numberOfIterations();
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    april_assert(N == span_it2.numberOfIterations());
    april_assert(size == static_cast<unsigned int>(span_it2.getSize()));
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    functor(m1, m2, size, stride1, stride2,
	    static_cast<unsigned int>(span_it1.getOffset()),
	    static_cast<unsigned int>(span_it2.getOffset()));
    if (N > 1) {
#ifndef NO_OMP
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
#endif
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
#ifndef NO_OMP
      }
#endif
    }
  }
}

// Idem but for binary functions (needs two matrices, and two
// best_span_iterators)
template<typename T, typename FUNC>
void applyBinaryFunctionWithSpanIterator(Matrix<T> *m1,
					 const Matrix<T> *m2,
					 const FUNC &functor,
					 const int N_th = DEFAULT_N_TH,
					 const unsigned int SIZE_th = DEFAULT_SIZE_TH) {
#ifdef NO_OMP
  UNUSED_VARIABLE(N_th);
  UNUSED_VARIABLE(SIZE_th);
#endif
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      m1->getIsDataRowOrdered() == m2->getIsDataRowOrdered())
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
    typename Matrix<T>::best_span_iterator span_it1(m1), span_it2(m2);
    const int N = span_it1.numberOfIterations();
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    april_assert(N == span_it2.numberOfIterations());
    april_assert(size == static_cast<unsigned int>(span_it2.getSize()));
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    functor(m1, m2, size, stride1, stride2,
	    static_cast<unsigned int>(span_it1.getOffset()),
	    static_cast<unsigned int>(span_it2.getOffset()));
    if (N > 1) {
#ifndef NO_OMP
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
#endif
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
#ifndef NO_OMP
      }
#endif
    }
  }
}

// Similar to previous functions, but for sum-reduction operations
template<typename T, typename FUNC>
T applySumReductionWithSpanIterator(const Matrix<T> *m,
				    const FUNC &functor,
				    const int N_th = DEFAULT_N_TH,
				    const unsigned int SIZE_th = DEFAULT_SIZE_TH) {
#ifdef NO_OMP
  UNUSED_VARIABLE(N_th);
  UNUSED_VARIABLE(SIZE_th);
#endif
  // Contiguous memory block
  if (m->getIsContiguous()) {
    T aux = functor(m, static_cast<unsigned int>(m->size()), 1,
		    static_cast<unsigned int>(m->getOffset()));
    return aux;
  }
  // One dimension
  else if (m->getNumDim() == 1)
    return functor(m, static_cast<unsigned int>(m->size()),
		   static_cast<unsigned int>(m->getStrideSize(0)),
		   static_cast<unsigned int>(m->getOffset()));
  // General case
  else {
    typename Matrix<T>::best_span_iterator span_it(m);
    const int N = span_it.numberOfIterations();
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    T sum;
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    sum = functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
    if (N > 1) {
#ifndef NO_OMP
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
#endif
	// sequential code, with less overhead when updating iterator
	++span_it;
	do {
	  sum += functor(m, size, stride,
			 static_cast<unsigned int>(span_it.getOffset()));
	  ++span_it;
	} while(span_it != m->end_span_iterator());
#ifndef NO_OMP
      }
#endif
    }
    return sum;
  }
}

// Similar to previous functions, but for sum-reduction operations
template<typename T, typename FUNC>
T applySumReductionWithSpanIteratorNOPARALLEL(const Matrix<T> *m,
					      const FUNC &functor) {
  // Contiguous memory block
  if (m->getIsContiguous())
    return functor(m, static_cast<unsigned int>(m->size()), 1,
		   static_cast<unsigned int>(m->getOffset()));
  // One dimension
  else if (m->getNumDim() == 1)
    return functor(m, static_cast<unsigned int>(m->size()),
		   static_cast<unsigned int>(m->getStrideSize(0)),
		   static_cast<unsigned int>(m->getOffset()));
  // General case
  else {
    typename Matrix<T>::best_span_iterator span_it(m);
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    T sum;
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    sum = functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
    // sequential code, with less overhead when updating iterator
    ++span_it;
    do {
      sum += functor(m, size, stride,
		     static_cast<unsigned int>(span_it.getOffset()));
      ++span_it;
    } while(span_it != m->end_span_iterator());
    return sum;
  }
}

// Similar to previous functions, but for AND-reduction operations (binary)
template<typename T, typename FUNC>
bool applyBinaryAndReductionWithSpanIterator(const Matrix<T> *m1,
					     const Matrix<T> *m2,
					     const FUNC &functor,
					     const int N_th = DEFAULT_N_TH,
					     const unsigned int SIZE_th = DEFAULT_SIZE_TH) {
#ifdef NO_OMP
  UNUSED_VARIABLE(N_th);
  UNUSED_VARIABLE(SIZE_th);
#endif
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      m1->getIsDataRowOrdered() == m2->getIsDataRowOrdered())
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
    typename Matrix<T>::best_span_iterator span_it1(m1), span_it2(m2);
    const int N = span_it1.numberOfIterations();
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    april_assert(N == span_it2.numberOfIterations());
    april_assert(size == static_cast<unsigned int>(span_it2.getSize()));
    // forward application of functor, to force execution of memory copy from GPU
    // to PPAL or viceversa (if needed), avoiding race conditions on the following
    bool ret = functor(m1, m2, size, stride1, stride2,
		       static_cast<unsigned int>(span_it1.getOffset()),
		       static_cast<unsigned int>(span_it2.getOffset()));
    if (ret && N > 1) {
#ifndef NO_OMP
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
#endif
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
#ifndef NO_OMP
      }
#endif
    }
    return ret;
  }
}

// Similar to previous functions, but for general reduction operations (without
// OMP)
template<typename T, typename R, typename FUNC1, typename FUNC2>
R applyReductionWithSpanIteratorNOPARALLEL(const Matrix<T> *m,
					   FUNC1 &functor,
					   FUNC2 &reductor,
					   const R initial_value) {
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
    typename Matrix<T>::best_span_iterator span_it(m);
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    R red(initial_value);
    // sequential code
    while(span_it != m->end_span_iterator()) { 
      R result = functor(m, size, stride,
			 static_cast<unsigned int>(span_it.getOffset()));
      red = reductor(red, result);
      ++span_it;
    }
    return red;
  }
}

// Auxiliary function template which applies a given FUNC object ( implements
// operator() ) to all the elements of a Matrix, using the best_span_iterator,
// WITHOUT OMP
template<typename T, typename FUNC>
void applyFunctionWithSpanIteratorNOPARALLEL(Matrix<T> *m,
					     const FUNC &functor) {
  // Contiguous memory block
  if (m->getIsContiguous())
    functor(m, static_cast<unsigned int>(m->size()), 1,
	    static_cast<unsigned int>(m->getOffset()));
  // One dimension
  else if (m->getNumDim() == 1)
    functor(m, static_cast<unsigned int>(m->size()),
	    static_cast<unsigned int>(m->getStrideSize(0)),
	    static_cast<unsigned int>(m->getOffset()));
  // General case
  else {
    typename Matrix<T>::best_span_iterator span_it(m);
    unsigned int size   = static_cast<unsigned int>(span_it.getSize());
    unsigned int stride = static_cast<unsigned int>(span_it.getStride());
    while(span_it != m->end_span_iterator()) {
      functor(m, size, stride, static_cast<unsigned int>(span_it.getOffset()));
      ++span_it;
    }
  }
}

// Idem but for binary functions (needs two matrices, and two
// best_span_iterators), NO OMP
template<typename T, typename FUNC>
void applyBinaryFunctionWithSpanIteratorNOPARALLEL(Matrix<T> *m1,
						   Matrix<T> *m2,
						   const FUNC &functor) {
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      m1->getIsDataRowOrdered() == m2->getIsDataRowOrdered())
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
    typename Matrix<T>::best_span_iterator span_it1(m1), span_it2(m2);
    const int N = span_it1.numberOfIterations();
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    april_assert(N == span_it2.numberOfIterations());
    april_assert(size == static_cast<unsigned int>(span_it2.getSize()));
    while(span_it1 != m1->end_span_iterator()) {
      functor(m1, m2, size, stride1, stride2,
	      static_cast<unsigned int>(span_it1.getOffset()),
	      static_cast<unsigned int>(span_it2.getOffset()));
      ++span_it1;
      ++span_it2;
    }
  }
}

// Idem but for binary functions (needs two matrices, and two
// best_span_iterators), NO OMP
template<typename T, typename FUNC>
void applyBinaryFunctionWithSpanIteratorNOPARALLEL(Matrix<T> *m1,
						   const Matrix<T> *m2,
						   const FUNC &functor) {
  if (m1->getIsContiguous() && m2->getIsContiguous() &&
      m1->getIsDataRowOrdered() == m2->getIsDataRowOrdered())
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
    typename Matrix<T>::best_span_iterator span_it1(m1), span_it2(m2);
    unsigned int size    = static_cast<unsigned int>(span_it1.getSize());
    unsigned int stride1 = static_cast<unsigned int>(span_it1.getStride());
    unsigned int stride2 = static_cast<unsigned int>(span_it2.getStride());
    april_assert(span_it1.numberOfIterations() == span_it2.numberOfIterations());
    april_assert(size == static_cast<unsigned int>(span_it2.getSize()));
    while(span_it1 != m1->end_span_iterator()) {
      functor(m1, m2, size, stride1, stride2,
	      static_cast<unsigned int>(span_it1.getOffset()),
	      static_cast<unsigned int>(span_it2.getOffset()));
      ++span_it1;
      ++span_it2;
    }
  }
}

#endif // MATRIX_GENERIC_MATH_TEMPLATES_H
