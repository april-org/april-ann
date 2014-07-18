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

template <typename T> typename Matrix<T>::sliding_window*
prepareMatrixSlideWindowConvolution(int D, Matrix<T> *mat,
                                    // step is an array with D size
                                    const int *step,
                                    // kernel is an array with D+2 size
                                    const int *kernel) {
  typename Matrix<T>::sliding_window *mat_sw;
  int numDim = mat->getNumDim();
  int *aux_step = 0;
  if (step != 0) {
    aux_step = new int[D+2];
    aux_step[0] = aux_step[1] = 1;
    for (int i=0; i<D; ++i) aux_step[i+2] = step[i];
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
                                                    (aux_step+2) : 0);
    break;
  case 1: // numDim == D+1
    // KxPxN1xN2x...xNd kernel :: PxM1xM2x...xMd matrix
    if (kernel[1] != mat->getDimSize(0)) {
      ERROR_EXIT2(128, "Incorrect kernel size at 2nd dimension, "
                  "expected %d, found %d\n", mat->getDimSize(0), kernel[1]);
    }
    mat_sw = new typename Matrix<T>::sliding_window(mat, kernel+1,
                                                    (step != 0) ?
                                                    (aux_step+1) : 0);
    break;
  case 0: // numDim == D+2
    {
      // KxPxN1xN2x...xNd kernel :: BxPxM1xM2x...xMd matrix
      if (kernel[1] != mat->getDimSize(1)) {
        ERROR_EXIT2(128, "Incorrect kernel size at 2nd dimension, "
                    "expected %d, found %d\n", mat->getDimSize(1), kernel[1]);
      }
      int *aux_kernel = new int[numDim];
      aux_kernel[0] = 1;
      for (int i=1; i<numDim; ++i) aux_kernel[i] = kernel[i];
      mat_sw = new typename Matrix<T>::sliding_window(mat, aux_kernel, aux_step);
      delete[] aux_kernel;
    }
    break;
  default:
    mat_sw = 0;
    ERROR_EXIT4(128, "Incorrect number of dimensions, expected "
                "%d or %d or %d, found %d\n", D, D+1, D+2, numDim);
  }
  delete[] aux_step;
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
  Matrix<T> *unrolled_slice = 0;
  // allocate unrolled matrix
  int dims[2] = { mat_sw->numWindows(), mat_slice->size() };
  Matrix<T> *unrolled_mat = new Matrix<T>(2, dims, mat->getMajorOrder());
  int *rewrapped_dims = new int[mat_slice->getNumDim()+1];
  rewrapped_dims[0] = dims[0];
  for (int i=0; i<mat_slice->getNumDim(); ++i) {
    rewrapped_dims[i+1] = mat_slice->getDimSize(i);
  }
  Matrix<T> *unrolled_mat_rewrapped =
    unrolled_mat->rewrap(rewrapped_dims, mat_slice->getNumDim()+1);
  delete[] rewrapped_dims;
  IncRef(unrolled_mat_rewrapped);
  typename Matrix<T>::sliding_window *unrolled_sw =
    new typename Matrix<T>::sliding_window(unrolled_mat_rewrapped);
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
                        !mat_slice->getIsContiguous());
    unrolled_slice->copy(rewrapped_mat_slice);
    delete rewrapped_mat_slice;
    //
    mat_sw->next();
    unrolled_sw->next();
  }
  delete mat_slice;
  delete unrolled_slice;
  delete mat_sw;
  delete unrolled_sw;
  DecRef(unrolled_mat_rewrapped);
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
    if (!result->sameDim(result_sizes, D+2)) {
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
  // check all matrices are contiguous
  if (!this->getIsContiguous() || !kernel->getIsContiguous()) {
    ERROR_EXIT(128, "Needs contiguous matrices\n");
  }
  int bunch_size = (D+2 == getNumDim()) ? getDimSize(0) : 1;
  // compute kernel sizes and unroll kernel matrix
  int *kernel_sizes;
  Matrix<T> *unrolled_kernel = unrollKernelForConvolution(D, kernel,
                                                          kernel_sizes);
  // allocate result to properly fit the convolution
  result = allocateResultMatrix(D, bunch_size, this,
                                (kernel_sizes == 0) ? kernel->getDimPtr() :
                                kernel_sizes,
                                step, result);
  // unroll all convolutions in one matrix, so the whole convolution will be
  // performed as a matrix-matrix multiplication
  Matrix<T> *unrolled_this =
    unrollSourceMatrixForConvolution(D, this, step,
                                     (kernel_sizes == 0) ? kernel->getDimPtr() :
                                     kernel_sizes);
  // unroll the result matrix
  int dims[2] = { unrolled_kernel->getDimSize(0),
                  unrolled_this->getDimSize(0) };
  Matrix<T> *unrolled_result = result->rewrap(dims, 2);
  // compute convolution by using CBLAS GEMM
  unrolled_result->gemm(CblasNoTrans, CblasTrans,
                        T(1.0f), unrolled_kernel, unrolled_this,
                        T(0.0f));
  //
  delete[] kernel_sizes;
  delete unrolled_this;
  delete unrolled_result;
  delete unrolled_kernel;
  return result;
}
