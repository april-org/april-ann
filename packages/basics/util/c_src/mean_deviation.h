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
#ifndef MEAN_DEVIATION_H
#define MEAN_DEVIATION_H

#include <cmath>

namespace april_utils {

  template<typename T>
    void mean_deviation(T v[], int sz, double *mean, double *dev) {
    double sum = 0, sum2 = 0;
    for (int i=0;i<sz;++i) {
      sum  += v[i];
      sum2 += v[i]*v[i];
    }
    double m = *mean = sum/sz;
    *dev  = sqrt(sum2/sz - m*m);
  }

  // http://www.johndcook.com/standard_deviation.html

  // This better way of computing variance goes back to a 1962 paper
  // by B. P. Welford and is presented in Donald Knuth's Art of
  // Computer Programming, Vol 2, page 232, 3rd edition. Although this
  // solution has been known for decades, not enough people know about
  // it. Most people are probably unaware that computing sample
  // variance can be difficult until the first time they compute a
  // standard deviation and get an exception for taking the square
  // root of a negative number.


  class RunningStat {
    int m_n;
    double m_oldM, m_newM, m_oldS, m_newS;

  public:
    RunningStat() : m_n(0) {}
    
    void Clear() {
      m_n = 0;
    }
    
    void Push(double x) {
      m_n++;
      
      // See Knuth TAOCP vol 2, 3rd edition, page 232
      if (m_n == 1) {
	  m_oldM = m_newM = x;
	  m_oldS = 0.0;
      } else {
	m_newM = m_oldM + (x - m_oldM)/m_n;
	m_newS = m_oldS + (x - m_oldM)*(x - m_newM);
	
	// set up for next iteration
	m_oldM = m_newM; 
	m_oldS = m_newS;
      }
    }
    
    int NumDataValues() const {
      return m_n;
    }
    
    double Mean() const {
      return (m_n > 0) ? m_newM : 0.0;
    }
    
    double Variance() const {
      return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0 );
    }
    
    double StandardDeviation() const {
      return sqrt( Variance() );
    }
    
  };

}

#endif //MEAN_DEVIATION_H
