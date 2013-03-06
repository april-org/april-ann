#ifndef IMAGE_METRICS_H
#define IMAGE_METRICS_H

#include "referenced.h"
#include "datasetFloat.h"

//Object that contains the counters for calculate the metrics
//
//The counters are doubles, but they can be used as integer (binary classification problem) or doubles (real values classification)
class ImageMetrics : public Referenced {
public:
  double true_positives;
  double false_positives;
  double true_negatives;
  double false_negatives;

  //
  double SSE;
  long  long int n_samples;
           
  ImageMetrics(double tp=0.0f, double fp=0.0f, double tn=0.0f, double fn=0.0f,
		double SSE=0.0, long int n_samples=0) : 
    true_positives(tp),
    false_positives(fp),
    true_negatives(tn),
    false_negatives(fn),
    SSE(SSE), n_samples(n_samples) {
  }

  //add one sample to the list
  void process_sample(float pred, float ref);
  void process_dataset(DataSetFloat *ds, DataSetFloat *GT, bool binary, float threshold);
            
  //retrieve metrics
  void get_metrics(double &FM, double &PR, double &RC, double &GA,double &MSE, double &TNR, double &ACC, double &PSNR, double &BRP, double &BRT, double &FNR);

  //Combining metrics
  void combine(ImageMetrics &m1);
  
  int nSamples();
  void clear();
  ImageMetrics* clone();

};

#endif
