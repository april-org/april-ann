/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
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
#ifndef MATRIX_CONV_IMPL_H
#define MATRIX_CONV_IMPL_H

#include "matrix.h"
#include "smart_ptr.h"

namespace basics {
  
  namespace MatrixExt {

    template <typename T>
    typename Matrix<T>::sliding_window *
    prepareMatrixSlideWindowConvolution(int D, Matrix<T> *mat,
                                        // step is an array with D size
                                        const int *step,
                                        // kernel is an array with D+2 size
                                        const int *kernel) {
      typename Matrix<T>::sliding_window *mat_sw=0;
      int numDim = mat->getNumDim();
      april_utils::UniquePtr<int []> aux_step, order_step;
      if (step != 0) {
        aux_step = new int[D+2];
        aux_step[0] = aux_step[1] = 1;
        for (int i=0; i<D; ++i) aux_step[i+2] = step[i];
      }
      if (mat->getMajorOrder() == CblasColMajor) {
        order_step = new int[numDim];
        for (int i=0; i<numDim; ++i) order_step[i] = i;
      }
      switch(D+2 - numDim) {
      case 2: // numDim == D
        // Kx1xN1xN2x...xNd kernel :: M1xM2x...xMd matrix
        if (kernel[1] != 1) {
          ERROR_EXIT1(128, "Incorrect kernel size at 2nd dimension, "
                      "expected 1, found %d\n", kernel[1]);
        }
        mat_sw = new typename Matrix<T>::sliding_window(mat, kernel+2,
                                                        (step != 0) ?
                                                        (aux_step.get()+2) : 0,
                                                        0, 0,
                                                        order_step.get());
        break;
      case 1: // numDim == D+1
        // KxPxN1xN2x...xNd kernel :: PxM1xM2x...xMd matrix
        if (kernel[1] != mat->getDimSize(0)) {
          ERROR_EXIT2(128, "Incorrect kernel size at 2nd dimension, "
                      "expected %d, found %d\n", mat->getDimSize(0), kernel[1]);
        }
        mat_sw = new typename Matrix<T>::sliding_window(mat, kernel+1,
                                                        (step != 0) ?
                                                        (aux_step.get()+1) : 0,
                                                        0, 0,
                                                        order_step.get());
        break;
      case 0: // numDim == D+2
        {
          // KxPxN1xN2x...xNd kernel :: BxPxM1xM2x...xMd matrix
          if (kernel[1] != mat->getDimSize(1)) {
            ERROR_EXIT2(128, "Incorrect kernel size at 2nd dimension, "
                        "expected %d, found %d\n", mat->getDimSize(1), kernel[1]);
          }
          april_utils::UniquePtr<int []> aux_kernel(new int[numDim]);
          aux_kernel[0] = 1; // mat->getDimSize(0);
          for (int i=1; i<numDim; ++i) aux_kernel[i] = kernel[i];
          mat_sw = new typename Matrix<T>::sliding_window(mat, aux_kernel.get(),
                                                          aux_step.get(),
                                                          0, 0,
                                                          order_step.get());
        }
        break;
      default:
        ERROR_EXIT4(128, "Incorrect number of dimensions, expected "
                    "%d or %d or %d, found %d\n", D, D+1, D+2, numDim);
      }
      return mat_sw;
    }

    // Traverses mat using a sliding_window configured to fit the given convolution
    // kernel, and copies every window into an unrolled matrix.
    template <typename T>
    Matrix<T> *unrollSourceMatrixForConvolution(int D, Matrix<T> *mat,
                                                const int *step,
                                                const int *kernel,
                                                int bunch_size) {
      april_utils::SharedPtr< typename Matrix<T>::sliding_window >
        mat_sw( prepareMatrixSlideWindowConvolution(D, mat, step, kernel) );
      april_utils::SharedPtr< Matrix<T> > mat_slice = mat_sw->getMatrix();
      april_utils::SharedPtr< Matrix<T> > unrolled_slice;
      // allocate unrolled matrix
      int dims[3] = { bunch_size,
                      mat_sw->numWindows()/bunch_size,
                      mat_slice->size() };
      Matrix<T> *unrolled_mat = new Matrix<T>(3, dims, mat->getMajorOrder());
      april_utils::UniquePtr<int []> aux_dims(new int[mat_slice->getNumDim()+2]);
      aux_dims[0] = dims[0];
      aux_dims[1] = dims[1];
      for (int i=0; i<mat_slice->getNumDim(); ++i) {
        aux_dims[i+2] = mat_slice->getDimSize(i);
      }
      april_utils::SharedPtr< Matrix<T> >
        unrolled_mat_rewrapped( unrolled_mat->rewrap(aux_dims.get(),
                                                     mat_slice->getNumDim()+2) );
      aux_dims[0] = 1;
      aux_dims[1] = 1;
      april_utils::UniquePtr<int []> order_step;
      if (mat->getMajorOrder() == CblasColMajor) {
        order_step = new int[mat_slice->getNumDim()+2];
        for (int i=0; i<mat_slice->getNumDim()+2; ++i) order_step[i] = i;
      }
      april_utils::SharedPtr< typename Matrix<T>::sliding_window >
        unrolled_sw( new typename Matrix<T>::
                     sliding_window(unrolled_mat_rewrapped.get(),
                                    aux_dims.get(), 0, 0, 0,
                                    order_step.get()) );
      if (unrolled_sw->numWindows() != mat_sw->numWindows()) {
        ERROR_EXIT(128, "Incorrect size in input matrix\n");
      }
      // copy all the slices
      while(!mat_sw->isEnd()) {
        april_assert( !mat_sw->isEnd() && !unrolled_sw->isEnd() );
        mat_slice      = mat_sw->getMatrix(mat_slice.get());
        unrolled_slice = unrolled_sw->getMatrix(unrolled_slice.get());
        //
        april_utils::SharedPtr< Matrix<T> >
          rewrapped_mat_slice( mat_slice->rewrap(unrolled_slice->getDimPtr(),
                                                 unrolled_slice->getNumDim(),
                                                 // clone in case it is not contiguous
                                                 true) );
        unrolled_slice->copy(rewrapped_mat_slice.get());
        //
        mat_sw->next();
        unrolled_sw->next();
      }
      return unrolled_mat;
    }

    template<typename T>
    Matrix<T> *unrollKernelForConvolution(int D, Matrix<T> *kernel,
                                          int *&kernel_sizes) {
      int dims[2];
      kernel_sizes = 0;
      switch(D+2 - kernel->getNumDim()) {
      case 0: // kernel->getNumDim() == D+2
        dims[0] = kernel->getDimSize(0);
        break;
      case 1: // kernel->getNumDim() == D+1
        kernel_sizes = new int[D+2];
        dims[0] = kernel_sizes[0] = 1;
        for (int i=0; i<D+1; ++i) kernel_sizes[i+1] = kernel->getDimSize(i);
        break;
      case 2: // kernel->getNumDim() == D
        kernel_sizes = new int[D+2];
        dims[0] = kernel_sizes[0] = kernel_sizes[1] = 1;
        for (int i=0; i<D; ++i) kernel_sizes[i+2] = kernel->getDimSize(i);
        break;
      default:
        ERROR_EXIT4(128,
                    "Incorrect kernel numDim, expected %d or %d or %d, given %d\n",
                    D, D+1, D+2, kernel->getNumDim());
      }
      dims[1] = kernel->size() / dims[0];
      Matrix<T> *unrolled_kernel = kernel->rewrap(dims, 2);
      return unrolled_kernel;
    }

    template<typename T>
    Matrix<T> *allocateResultMatrix(int D, int bunch_size,
                                    Matrix<T> *mat,
                                    const int *kernel_sizes,
                                    const int *step,
                                    Matrix<T> *result) {
      // compute result_sizes
      april_utils::UniquePtr<int []> result_sizes(new int[D+2]);
      result_sizes[0] = bunch_size;
      result_sizes[1] = kernel_sizes[0];
      int j = mat->getNumDim() - D; // first mat dimension
      if (step != 0) {
        // with a given step array
        for (int i=0; i<D; ++i) {
          result_sizes[i+2] = (mat->getDimSize(j+i) - kernel_sizes[i+2])/step[i] + 1;
        }
      }
      else {
        // without a given step array, assumed as step[i]=1
        for (int i=0; i<D; ++i) {
          result_sizes[i+2] = (mat->getDimSize(j+i) - kernel_sizes[i+2]) + 1;
        }
      }
      // allocate new result matrix if not given
      if (result == 0) {
        result = new Matrix<T>(D+2, result_sizes.get(), mat->getMajorOrder());
      }
      // check result matrix sizes (includes number of planes = number of kernels)
      else {
        if (!result->getIsContiguous()) {
          ERROR_EXIT(128, "Needs contiguous matrices\n");
        }
        if (!result->sameDim(result_sizes.get(), D+2)) {
          ERROR_EXIT(128, "Incorrect result matrix sizes\n");
        }
      }
      return result;
    }

    // The matrix must be of BxPxM1xM2x...xMd sizes, where B is the bunch_size, P
    // the number of planes. The kernel is matrix of KxPxN1xN2x...xNd sizes, where K
    // is the number of kernels, P the number of planes at input matrix. In both
    // cases, the left-most dimensions can be removed in order and them will be
    // taken as 1, that is, it is possible to remove B and P (or K and P), or to
    // remove only B (or K). In any case, the resulting matrix will be of
    // BxKxR1xR2x...xRd, and if given, it can ignore the two left-most dimensions.
    template <typename T>
    Matrix<T> *matConvolution(Matrix<T> *obj,
                              int D, const int *step,
                              Matrix<T> *kernel, Matrix<T> *result) {
      /*
        Matrix<T> **given_unrolled_kernel,
        Matrix<T> **given_unrolled_input) {
      */
      // check kernel matrix is contiguous
      if (!kernel->getIsContiguous()) {
        ERROR_EXIT(128, "Needs a contiguous kernel matrix\n");
      }
      int bunch_size = (D+2 == obj->getNumDim()) ? obj->getDimSize(0) : 1;
      // compute kernel sizes and unroll kernel matrix
      int *kernel_sizes;
      april_utils::SharedPtr< Matrix<T> > unrolled_kernel;
      unrolled_kernel = unrollKernelForConvolution(D, kernel, kernel_sizes);
      // allocate result to properly fit the convolution
      result = allocateResultMatrix(D, bunch_size, obj,
                                    (kernel_sizes == 0) ? kernel->getDimPtr() :
                                    kernel_sizes,
                                    step, result);
      // unroll all convolutions in one matrix, so the whole convolution will be
      // performed as a matrix-matrix multiplication
      april_utils::SharedPtr< Matrix<T> >
        unrolled_this( unrollSourceMatrixForConvolution(D, obj, step,
                                                        (kernel_sizes == 0) ? kernel->getDimPtr() :
                                                        kernel_sizes,
                                                        bunch_size) );
      // unroll the result matrix
      int dims[3] = { bunch_size,
                      unrolled_kernel->getDimSize(0),
                      unrolled_this->getDimSize(1) };
      april_utils::SharedPtr< Matrix<T> > unrolled_result;
      unrolled_result = result->rewrap(dims, 3);
      //
      april_utils::SharedPtr< Matrix<T> > source_pattern, result_pattern;
      source_pattern = unrolled_this->select(0, 0);
      result_pattern = unrolled_result->select(0, 0);
      april_utils::SharedPtr< Matrix<T> > contiguous_source_pattern, contiguous_result_pattern;
      contiguous_source_pattern = ( (source_pattern->getIsContiguous()) ?
                                    source_pattern :
                                    source_pattern->clone() );
      contiguous_result_pattern = ( (result_pattern->getIsContiguous()) ?
                                    result_pattern :
                                    result_pattern->clone() );
      // compute convolution by using CBLAS GEMM
      contiguous_result_pattern->gemm(CblasNoTrans, CblasTrans,
                                      T(1.0f),
                                      unrolled_kernel.get(),
                                      contiguous_source_pattern.get(),
                                      T(0.0f));
      result_pattern->copy(contiguous_result_pattern.get());
      for (int j=1; j<bunch_size; ++j) {
        source_pattern = unrolled_this->select(0, j, source_pattern.get());
        result_pattern = unrolled_result->select(0, j, result_pattern.get());
        contiguous_source_pattern->copy(source_pattern.get());
        contiguous_result_pattern->copy(result_pattern.get());
        // compute convolution by using CBLAS GEMM
        contiguous_result_pattern->gemm(CblasNoTrans, CblasTrans,
                                        T(1.0f),
                                        unrolled_kernel.get(),
                                        contiguous_source_pattern.get(),
                                        T(0.0f));
        result_pattern->copy(contiguous_result_pattern.get());
      }
      delete[] kernel_sizes;
      return result;
    }

    // TODO: FINISH THE FOLLOWING EXPERIMENTAL IMPLEMENTATION

    /*
      template <typename T> typename Matrix<T>::sliding_window*
      prepareMatrixSlideWindowConvolution(int D, Matrix<T> *mat,
      // step is an array with D size
      const int *step,
      // kernel is an array with D+2 size
      const int *kernel) {
      typename Matrix<T>::sliding_window *mat_sw;
      int numDim = mat->getNumDim();
      int *aux_step = 0;
      int *order_step = 0;
      if (step != 0) {
      aux_step = new int[D+2];
      aux_step[0] = aux_step[1] = 1;
      for (int i=0; i<D; ++i) aux_step[i+2] = step[i];
      }
      if (mat->getMajorOrder() == CblasColMajor) {
  
      //order_step = new int[numDim];
      //for (int i=0; i<numDim; ++i) order_step[i] = i;
  
      }
      switch(D+2 - numDim) {
      case 2: // numDim == D
      // Kx1xN1xN2x...xNd kernel :: M1xM2x...xMd matrix
      if (kernel[1] != 1) {
      ERROR_EXIT1(128, "Incorrect kernel size at 2nd dimension, "
      "expected 1, found %d\n", kernel[1]);
      }
      mat_sw = new typename Matrix<T>::sliding_window(mat, kernel+2,
      (step != 0) ?
      (aux_step+2) : 0,
      0, 0,
      order_step);
      break;
      case 1: // numDim == D+1
      // KxPxN1xN2x...xNd kernel :: PxM1xM2x...xMd matrix
      if (kernel[1] != mat->getDimSize(0)) {
      ERROR_EXIT2(128, "Incorrect kernel size at 2nd dimension, "
      "expected %d, found %d\n", mat->getDimSize(0), kernel[1]);
      }
      mat_sw = new typename Matrix<T>::sliding_window(mat, kernel+1,
      (step != 0) ?
      (aux_step+1) : 0,
      0, 0,
      order_step);
      break;
      case 0: // numDim == D+2
      {
      // KxPxN1xN2x...xNd kernel :: BxPxM1xM2x...xMd matrix
      if (kernel[1] != mat->getDimSize(1)) {
      ERROR_EXIT2(128, "Incorrect kernel size at 2nd dimension, "
      "expected %d, found %d\n", mat->getDimSize(1), kernel[1]);
      }
      int *aux_kernel = new int[numDim];
      aux_kernel[0] = 1; // mat->getDimSize(0);
      for (int i=1; i<numDim; ++i) aux_kernel[i] = kernel[i];
      mat_sw = new typename Matrix<T>::sliding_window(mat, aux_kernel,
      aux_step,
      0, 0,
      order_step);
      delete[] aux_kernel;
      }
      break;
      default:
      mat_sw = 0;
      ERROR_EXIT4(128, "Incorrect number of dimensions, expected "
      "%d or %d or %d, found %d\n", D, D+1, D+2, numDim);
      }
      delete[] aux_step;
      delete[] order_step;
      return mat_sw;
      }

      // Traverses mat using a sliding_window configured to fit the given convolution
      // kernel, and copies every window into an unrolled matrix.
      template <typename T>
      Matrix<T> *unrollSourceMatrixForConvolution(int D, Matrix<T> *mat,
      const int *step,
      const int *kernel) {
      typename Matrix<T>::sliding_window *mat_sw =
      prepareMatrixSlideWindowConvolution(D, mat, step, kernel);
      Matrix<T> *mat_slice      = mat_sw->getMatrix();
      IncRef(mat_slice);
      Matrix<T> *unrolled_slice = 0;
      // allocate unrolled matrix
      int dims[2] = { mat_sw->numWindows(),
      mat_slice->size() };
      Matrix<T> *unrolled_mat = new Matrix<T>(2, dims, mat->getMajorOrder());
      unrolled_mat->zeros();
      int *aux_dims = new int[mat_slice->getNumDim()+1];
      aux_dims[0] = dims[0];
      for (int i=0; i<mat_slice->getNumDim(); ++i) {
      aux_dims[i+1] = mat_slice->getDimSize(i);
      }
      Matrix<T> *unrolled_mat_rewrapped =
      unrolled_mat->rewrap(aux_dims, mat_slice->getNumDim()+1);
      IncRef(unrolled_mat_rewrapped);
      aux_dims[0] = 1;
      int *order_step = 0;
      if (mat->getMajorOrder() == CblasColMajor) {
      order_step = new int[mat_slice->getNumDim()+1];
      for (int i=0; i<mat_slice->getNumDim()+1; ++i) order_step[i] = i;
      }
      typename Matrix<T>::sliding_window *unrolled_sw =
      new typename Matrix<T>::sliding_window(unrolled_mat_rewrapped,
      aux_dims, 0, 0, 0,
      order_step);
      delete[] aux_dims;
      delete[] order_step;
      if (unrolled_sw->numWindows() != mat_sw->numWindows()) {
      ERROR_EXIT(128, "Incorrect size in input matrix\n");
      }
      // copy all the slices
      while(!mat_sw->isEnd()) {
      april_assert( !mat_sw->isEnd() && !unrolled_sw->isEnd() );
      mat_slice      = mat_sw->getMatrix(mat_slice);
      unrolled_slice = unrolled_sw->getMatrix(unrolled_slice);
      //
      Matrix<T> *rewrapped_mat_slice =
      mat_slice->rewrap(unrolled_slice->getDimPtr(),
      unrolled_slice->getNumDim(),
      // clone in case it is not contiguous
      true);
      IncRef(rewrapped_mat_slice);
      unrolled_slice->copy(rewrapped_mat_slice);
      //rewrapped_mat_slice->print();
      //unrolled_slice->print();
      //unrolled_mat_rewrapped->print();
      //unrolled_mat->print();
      //printf("========================================================\n");
      DecRef(rewrapped_mat_slice);
      //
      mat_sw->next();
      unrolled_sw->next();
      }
      delete unrolled_slice;
      delete unrolled_sw;
      DecRef(unrolled_mat_rewrapped);
      DecRef(mat_slice);
      delete mat_sw;
      //unrolled_mat->print();
      return unrolled_mat;
      }

      template<typename T>
      Matrix<T> *unrollKernelForConvolution(int D, Matrix<T> *kernel,
      int *&kernel_sizes) {
      int dims[2];
      kernel_sizes = 0;
      switch(D+2 - kernel->getNumDim()) {
      case 0: // kernel->getNumDim() == D+2
      dims[0] = kernel->getDimSize(0);
      break;
      case 1: // kernel->getNumDim() == D+1
      kernel_sizes = new int[D+2];
      dims[0] = kernel_sizes[0] = 1;
      for (int i=0; i<D+1; ++i) kernel_sizes[i+1] = kernel->getDimSize(i);
      break;
      case 2: // kernel->getNumDim() == D
      kernel_sizes = new int[D+2];
      dims[0] = kernel_sizes[0] = kernel_sizes[1] = 1;
      for (int i=0; i<D; ++i) kernel_sizes[i+2] = kernel->getDimSize(i);
      break;
      default:
      ERROR_EXIT4(128,
      "Incorrect kernel numDim, expected %d or %d or %d, given %d\n",
      D, D+1, D+2, kernel->getNumDim());
      }
      dims[1] = kernel->size() / dims[0];
      Matrix<T> *unrolled_kernel = kernel->rewrap(dims, 2);
      return unrolled_kernel;
      }

      template<typename T>
      Matrix<T> *allocateResultMatrix(int D, int bunch_size,
      Matrix<T> *mat,
      const int *kernel_sizes,
      const int *step,
      Matrix<T> *result) {
      // compute result_sizes
      int *result_sizes = new int[D+2];
      result_sizes[0] = bunch_size;
      result_sizes[1] = kernel_sizes[0];
      int j = mat->getNumDim() - D; // first mat dimension
      if (step != 0) {
      // with a given step array
      for (int i=0; i<D; ++i) {
      result_sizes[i+2] = (mat->getDimSize(j+i) - kernel_sizes[i+2])/step[i] + 1;
      }
      }
      else {
      // without a given step array, assumed as step[i]=1
      for (int i=0; i<D; ++i) {
      result_sizes[i+2] = (mat->getDimSize(j+i) - kernel_sizes[i+2]) + 1;
      }
      }
      // allocate new result matrix if not given
      if (result == 0) {
      result = new Matrix<T>(D+2, result_sizes, mat->getMajorOrder());
      }
      // check result matrix sizes (includes number of planes = number of kernels)
      else {
      if (!result->getIsContiguous()) {
      ERROR_EXIT(128, "Needs contiguous matrices\n");
      }
      if (!result->sameDim(result_sizes, D+1)) {
      ERROR_EXIT(128, "Incorrect result matrix sizes\n");
      }
      }
      delete[] result_sizes;
      return result;
      }

      // The matrix must be of BxPxM1xM2x...xMd sizes, where B is the bunch_size, P
      // the number of planes. The kernel is matrix of KxPxN1xN2x...xNd sizes, where K
      // is the number of kernels, P the number of planes at input matrix. In both
      // cases, the left-most dimensions can be removed in order and them will be
      // taken as 1, that is, it is possible to remove B and P (or K and P), or to
      // remove only B (or K). In any case, the resulting matrix will be of
      // BxKxR1xR2x...xRd, and if given, it can ignore the two left-most dimensions.
      template <typename T>
      Matrix<T> *Matrix<T>::convolution(int D, const int *step,
      Matrix<T> *kernel, Matrix<T> *result) {
      // TODO: make it work for CblasColMajor order
      if (getMajorOrder() == CblasColMajor) {
      ERROR_EXIT(128, "Not implemented for col_major matrices\n");
      }
      // Matrix<T> **given_unrolled_kernel,
      // Matrix<T> **given_unrolled_input) {
      // check kernel matrix is contiguous
      if (!kernel->getIsContiguous()) {
      ERROR_EXIT(128, "Needs a contiguous kernel matrix\n");
      }
      int bunch_size = (D+2 == getNumDim()) ? getDimSize(0) : 1;
      // compute kernel sizes and unroll kernel matrix
      int *kernel_sizes;
      Matrix<T> *unrolled_kernel =
      unrollKernelForConvolution(D, kernel, kernel_sizes);
      IncRef(unrolled_kernel);
      // allocate result to properly fit the convolution
      result = allocateResultMatrix(D, bunch_size, this,
      (kernel_sizes == 0) ? kernel->getDimPtr() :
      kernel_sizes,
      step, result);
      // unroll all convolutions in one matrix, so the whole convolution will be
      // performed as a matrix-matrix multiplication
      Matrix<T> *unrolled_input =
      unrollSourceMatrixForConvolution(D, this, step,
      (kernel_sizes == 0) ? kernel->getDimPtr() :
      kernel_sizes);
      IncRef(unrolled_input);
      // unroll the result matrix
      int dims[2];
      dims[0] = result->getDimSize(0) * result->getDimSize(1);
      dims[1] = result->size() / dims[0];
      Matrix<T> *unrolled_result = result->rewrap(dims, 2);
      IncRef(unrolled_result);
      //
      int sizes[2];
      sizes[0] = unrolled_input->getDimSize(0)/bunch_size;
      sizes[1] = unrolled_input->getDimSize(1);
      Matrix<T>::sliding_window source_sw(unrolled_input, sizes, 0, sizes);
      sizes[0] = unrolled_kernel->getDimSize(0);
      sizes[1] = unrolled_result->getDimSize(1);
      Matrix<T>::sliding_window result_sw(unrolled_result, sizes, 0, sizes);
      // sanity check
      april_assert(source_sw.numWindows() == bunch_size);
      april_assert(result_sw.numWindows() == bunch_size);
      //
      Matrix<T> *source_pattern = 0, *result_pattern = 0;
      source_pattern = source_sw.getMatrix();
      result_pattern = result_sw.getMatrix();
      IncRef(source_pattern);
      IncRef(result_pattern);
      // april_utils::aprilPrint(0, source_sw.isEnd(), "\n");
      while(!source_sw.isEnd()) {
      source_pattern = source_sw.getMatrix(source_pattern);
      result_pattern = result_sw.getMatrix(result_pattern);
      // compute convolution by using CBLAS GEMM
      result_pattern->gemm(CblasNoTrans, CblasTrans,
      T(1.0f),
      unrolled_kernel, source_pattern,
      T(0.0f));
      //
      //source_pattern->print();
      //result_pattern->print();
      //printf("========================================================\n");
      source_sw.next();
      result_sw.next();
      }
      DecRef(source_pattern);
      DecRef(result_pattern);
      DecRef(unrolled_input);
      DecRef(unrolled_result);
      DecRef(unrolled_kernel);
      delete[] kernel_sizes;
      return result;
      }
    */
  } // namespace MatrixExt
} // namespace basics

#endif // MATRIX_CONV_IMPL_H
