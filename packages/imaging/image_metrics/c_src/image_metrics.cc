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

#include "error_print.h"
#include "image_metrics.h"
#include <cmath>
void ImageMetrics::processSample(float pred, float ref){

  double act_tp;
  double act_fp;
  double act_tn;
  double act_fn;

  //tp  p(pred=black | ref = black)
  act_tp = (1-pred)*(1-ref);
  //fp  p(pred=black | ref = white)
  act_fp = (1-pred)*(ref);
  //fn  p(pred = white | ref = white)
  act_tn = (pred)*(ref);
  //tp  p(pred = white | ref = black)
  act_fn = (pred)*(1-ref);

  true_positives  += act_tp;
  false_positives += act_fp;
  true_negatives  += act_tn;
  false_negatives += act_fn;

  //SSE
  double act_sse = (pred-ref)*(pred-ref);
  SSE += act_sse;
  ++n_samples;

}

// TODO: Overload the + operator
void ImageMetrics::clear(){
    true_positives = 0.0;
    false_positives = 0.0;
    true_negatives = 0.0;
    false_negatives = 0.0;
    SSE = 0.0;
    n_samples = 0;
}

int ImageMetrics::nSamples(){
    return n_samples;
}

void ImageMetrics::combine(ImageMetrics &m1){
    true_positives += m1.true_positives;
    false_positives += m1.false_positives;
    true_negatives += m1.true_negatives;
    false_negatives += m1.false_negatives;

    SSE += m1.SSE;
    n_samples += m1.n_samples;
}

bool ImageMetrics::getMetrics(double &FM, double &PR, double &RC, double &GA,double &MSE, double &TNR, double &ACC, double &PSNR, double &BRP, double &BRR, double &FNR){

    if (n_samples == 0) {
        FM = PR = RC = GA = MSE = TNR = ACC = BRP = BRR = FNR = 0.0f;
        return false; 
    }

    PR = true_positives/(true_positives+false_positives);
    RC = true_positives/(true_positives+false_negatives);

    if(PR > 1 || RC > 1) {  
        fprintf(stderr,"TP: %f, FP: %f, TN: %f, FN: %f (%lld)\n",true_positives, false_positives, true_negatives, false_negatives,n_samples);
        exit(0);
    }
    if (true_positives == 0) 
        FM = 0.0;
    else
        FM = 2*PR*RC/(PR+RC);

    double b = (true_positives);
    double B = (true_positives+false_negatives);
    double w = (true_negatives);
    double W = (false_positives+true_negatives);
    
    // GA
    GA = sqrt((b*w)/(B*W));
    // TNR
    TNR = true_negatives/(true_negatives+false_positives);
    // FNR
    FNR = false_positives/(true_negatives+false_positives);
    // ACC
    ACC = (true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives);
    // MSE
    MSE = SSE/n_samples;
    // PSNR
    PSNR = 10 * log10(1./MSE);
    // BRP Black Ratio (Image Predicted)
    BRP = b/n_samples;
    // BRP Black Ratio (Reference)
    BRR = B/n_samples;
    
    return true;
}


void ImageMetrics::processDataset(DataSetFloat *ds, DataSetFloat *GT, bool binary, float threshold)
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
    const long int patSize = ds->patternSize();
    const long int numPats = ds->numPatterns();
    if (patSize != 1) {
        ERROR_PRINT("F measure: patternSize different of 1\n");
        exit(128);
    }
    //
    //
    float *pat_ds = new float[patSize], *pat_GT = new float[patSize];
    for (long int i=0; i<numPats; ++i) {
        ds->getPattern(i, pat_ds);
        GT->getPattern(i, pat_GT);

        /*        if ( pat_GT[0] > 1 || pat_GT[0] < 0)
                  fprintf(stderr, "Warning! Pattern Ground Truth value not in 0 ~ 1 range. Pat: %ld, value: %f\n",i, pat_GT[0]);

                  if ( pat_ds[0] > 1 || pat_ds[0] < 0)
                  fprintf(stderr, "Warning! Pattern predicted value not in 0 ~ 1 range. Pat: %ld, value: %f\n",i, pat_ds[0]);
                  */

        if (pat_GT[0] > 1) pat_GT[0] = 1.0;
        if (pat_GT[0] < 0) pat_GT[0] = 0.0;
        if (pat_ds[0] > 1) pat_ds[0] = 1.0;
        if (pat_ds[0] < 0) pat_ds[0] = 0.0;
        if (binary){
            pat_ds[0] = (pat_ds[0] <= threshold)? 0 : 1;    

            pat_GT[0] = (pat_GT[0] <= threshold)? 0 : 1;
        }
        processSample(pat_ds[0], pat_GT[0]);

    }
    delete[] pat_ds;
    delete[] pat_GT;
    return;
}

ImageMetrics* ImageMetrics::clone() {
    return new ImageMetrics(true_positives, false_positives,
            true_negatives, false_negatives,
            SSE, n_samples);
}






