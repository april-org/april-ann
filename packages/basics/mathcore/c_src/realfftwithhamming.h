/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2010, Salvador España-Boquera
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
#ifndef REALFFTWITHHAMMING_H
#define REALFFTWITHHAMMING_H

namespace AprilMath {

  /**
   * @brief Class for computation of real FFT with Hamming window.
   *
   * This class is intended to contain auxiliary work space to compute FFT
   * over \c double arrays. An object can be declared and used as many times as
   * needed.
   *
   * @code
   * const int input_size = 256;
   * AprilMath::RealFFTwithHamming fft(input_size);
   * const int output_size = fft.getOutputSize();
   * double *input = new double[input_size];
   * double *output = new double[output_size];
   * while(any condition) {
   *   takeInputData(input);
   *   fft(input, output);
   *   processOutputData(output);
   * }
   * delete[] input;
   * delete[] output;
   * @endcode
   *
   * @see Constructor RealFFTwithHamming::RealFFTwithHamming()
   *
   * @note This object is not thread safe because of the auxiliary work space.
   */
  class RealFFTwithHamming {
    /// Struct with cosine and sine auxilairy values.
    struct nodtblfft {
      double c,s,c3,s3;
    };
    int vSize,   ///< Size of the input vector
      nFFT,      ///< Number of FFT bins computed for a @c vSize input vector.
      bitsnFFT;  ///< Number of bits, logarithm in base 2 of @c nFFT .
    double *vec, ///< Auxiliary vector for input transformation (Hamming + padding zeros).
      *tmp,      ///< Auxiliary work space for FFT.
      *hamming_window; ///< Hamming window filter.
    nodtblfft *tbl;  ///< Table with cosine and sine auxiliary values.
    // TODO: This property can be promoted to a constant.
    double invsqrt2; ///< @c 1/sqrt(2)
  public:
    /**
     * @brief Constructs the object for input vectors with @c vSize length.
     *
     * @param vSize - The input vector size.
     *
     * @note If the @c vSize parameter is not a power of two, the input vector
     * would be zero padded to the closest power of two.
     */
    RealFFTwithHamming(int vSize);
    ~RealFFTwithHamming();
    /// Returns the number of FFT bins computed for the given input vector size.
    int getOutputSize() const { return nFFT/2; }
    /// Given an @c input, computes the FFT transformation in @c output .
    void operator() (const double input[], double output[]);
  };

}

#endif // REALFFTWITHHAMMING
