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

  class RealFFTwithHamming {
    struct nodtblfft {
      double c,s,c3,s3;
    };
    int vSize,nFFT,bitsnFFT;
    double *vec,*tmp,*hamming_window;
    nodtblfft *tbl;
    double invsqrt2;
  public:
    RealFFTwithHamming(int vSize);
    ~RealFFTwithHamming();
    int getOutputSize() const { return nFFT/2; }
    void operator() (const double input[], double output[]);
  };

}

#endif // REALFFTWITHHAMMING
