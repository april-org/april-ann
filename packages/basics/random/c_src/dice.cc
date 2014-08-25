/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
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
#include "dice.h"

namespace basics {

  Dice::Dice(int outcom, double *prob) {
    outcomes = outcom;
    threshold = new double[outcomes];
    double sum=0.0;
    for (int i=0; i<outcomes; i++) {
      sum += prob[i];
      threshold[i] = sum;
    }
    sum = 1.0/sum;
    for (int i=0; i<outcomes-1; i++)
      threshold[i] *= sum;
  }
  Dice::~Dice() {
    delete[] threshold;
  }
  int Dice::thrown(MTRand *generator) {
    double key = generator->rand(); //real number in [0,1]
    int left=0,right=outcomes-1;
    while (left < right) {
      int middle = (left+right)/2;
      if (threshold[middle] <= key) 
        left = middle+1;
      else
        right = middle;
    }
    return left;
  };

} // namespace basics
