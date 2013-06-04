/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco
 * Zamora-Martinez, Joan Pastor-Pellicer
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
#ifndef IMAGE_METRICS_H
#define IMAGE_METRICS_H

#include "referenced.h"
#include "datasetFloat.h"

/**
 Class that contains the counters for calculate the metrics on two different
 datasets (predicted and reference)

 The counters are doubles, but they can be used as integer (binary classification problem)
 or doubles (real values classification)
**/
class ImageMetrics : public Referenced {
public:

  /// Classification Counters
  double true_positives;
  double false_positives;
  double true_negatives;
  double false_negatives;
  
  //// The SSE cannot be computed throw the counters
  double SSE;
  long  long int n_samples;
           
  //// Creator
  ImageMetrics(double tp=0.0f, double fp=0.0f, double tn=0.0f, double fn=0.0f,
		double SSE=0.0, long int n_samples=0) : 
    true_positives(tp),
    false_positives(fp),
    true_negatives(tn),
    false_negatives(fn),
    SSE(SSE), n_samples(n_samples) {
  }

  //// Add one sample to the counters
  void processSample(float pred, float ref);

  //// Takes two datasets and add the information to the counters
  void processDataset(DataSetFloat *ds, DataSetFloat *GT, bool binary, float threshold);
            
  /** Returns differents measures
    FM   - Fmeasure
    PR   - Precission
    RC   - Recall
    GA   - Geometrical Arithmetic 
    TNR  - True Negative Rate
    PSNR - Peak signal-to-noise ratio
    ACC  - Acurracy
    BRP  - Black Rate Predicted (ratio of black pixels in prediction)
    BRR  - Black Rate Reference
  **/
  bool getMetrics(double &FM, double &PR, double &RC, double &GA, double &MSE, double &TNR, double &ACC, double &PSNR, double &BRP, double &BRR, double &FNR);

  //// Take a classes and add the counters to this
  void combine(ImageMetrics &m1);
  
  //// Return the total number of samples
  int nSamples();

  //// Reset the counters
  void clear();

  ImageMetrics* clone();
};

#endif
