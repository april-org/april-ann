/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Francisco Zamora-Martinez, Jorge
 * Gorbe-Moya
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
#include "error_print.h"
#include "fmeasure.h"

float Fmeasure(DataSetFloat *ds, DataSetFloat *GT,
	       float &PR,
	       float &RC)
{
  // Error control
  if (ds->numPatterns() != GT->numPatterns()) {
    ERROR_PRINT2("F measure: diferent numPatterns(): %d != %d\n",
		 ds->numPatterns(),
		 GT->numPatterns());
    exit(128);
  }
  if (ds->patternSize() != GT->patternSize()) {
    ERROR_PRINT2("F measure: diferent patternSize(): %d != %d\n",
		 ds->patternSize(),
		 GT->patternSize());
    exit(128);
  }
  if (ds->patternSize() != 1) {
    ERROR_PRINT1("F measure: patternSize must be 1, and is %d\n",
		 ds->patternSize());
    exit(128);
  }
  ///////////////////////////////////////////////////////////////
  const int patSize = ds->patternSize();
  const int numPats = ds->numPatterns();
  if (patSize != 1) {
    ERROR_PRINT("F measure: patternSize different of 1\n");
    exit(128);
  }
  //
  int Ctp=0; // number of true positives
  int Cfp=0; // number of false positives
  int Cfn=0; // number of false negatives
  //
  float *pat_ds = new float[patSize], *pat_GT = new float[patSize];
  for (int i=0; i<numPats; ++i) {
    ds->getPattern(i, pat_ds);
    GT->getPattern(i, pat_GT);
    if (positive(pat_ds[0])) {
      // true positive
      if (positive(pat_GT[0])) ++Ctp;
      // false positive
      else ++Cfp;
    }
    // false negative
    else if (positive(pat_GT[0])) ++Cfn;
  }
  //
  RC = float(Ctp)/(Cfn + Ctp);
  PR = float(Ctp)/(Cfp + Ctp);
  float FM = (2*RC*PR / (RC+PR)) * 100;
  delete[] pat_ds;
  delete[] pat_GT;
  return FM;
}
