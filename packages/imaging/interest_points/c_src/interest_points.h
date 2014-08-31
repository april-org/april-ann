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

#ifndef INTEREST_POINTS_H
#define INTEREST_POINTS_H

#include <cmath>
#include "image.h"
#include "image_connected_components.h"
#include "datasetFloat.h"
#include "utilImageFloat.h"
#include "pair.h"
#include "vector.h"
#include "geometry.h"

using AprilUtils::vector;
using AprilUtils::Point2D;
namespace InterestPoints
{
  
  enum {ASCENDER=1, TOPLINE, CENTRAL, BASELINE, DESCENDER, OTHER};
  /**
   * @brief Given an Image returns a vector with the local maxima and local minima of the given image.
   *
   * @param[in] pimg Pointer to the image
   * @param[in] threshold_white More than this value is considered black
   * @param[in] threshold_black Less than this value is set to white
   * @param[in] local_context The number of pixels in stroke that is used to compute the local maxima/minima
   * @param[in] duplicate_interval The minimum distance of locals within the same stroke
   */
  AprilUtils::vector<Point2D>* extract_points_from_image_old(Imaging::ImageFloat *pimg, float threshold_white = 0.6, float threshold_black = 0.4, int local_context = 6, int duplicate_interval = 5);

  /**
   * @brief Given an Image returns a vector with the local maxima and local minima of the given image.
   *
   * @param[in] pimg Pointer to the image
   * @param[out] local_minima Return the points of the local minima contour
   * @param[out] local_maxima Return the points of the local a contour
   * @param[in] threshold_white More than this value is considered black
   * @param[in] threshold_black Less than this value is set to white
   * @param[in] local_context The number of pixels in stroke that is used to compute the local maxima/minima
   * @param[in] duplicate_interval The minimum distance of locals within the same stroke
   */
  void extract_points_from_image(Imaging::ImageFloat *pimg, AprilUtils::vector<Point2D> *local_minima, AprilUtils::vector<Point2D> *local_maxima, float threshold_white= 0.4, float threshold_black = 0.6, int local_context = 6, int duplicate_interval = 3, bool reverse = true);


  Imaging::ImageFloatRGB * area_to_rgb(Imaging::ImageFloat *img);
  /*
   *
   *
   *
   */
  Imaging::ImageFloat *get_pixel_area(Imaging::ImageFloat *source,
                                      AprilUtils::vector<Point2D> ascenders,
                                      AprilUtils::vector<Point2D> upper_baseline, 
                                      AprilUtils::vector<Point2D> lower_baseline,
                                      AprilUtils::vector<Point2D> descenders,
                                      Basics::MatrixFloat **transitions);

  Basics::MatrixFloat *get_image_matrix_from_index(Basics::DataSetFloat *ds_out,
                                                   Basics::DataSetFloat *indexed,
                                                   int width,
                                                   int height,
                                                   int num_classes = 3);

  Imaging::ImageFloat *get_image_area_from_dataset(Basics::DataSetFloat *ds_out, Basics::DataSetFloat *indexed, int width, int height, int num_classes, float threshold = 0.8); 
  Basics::MatrixFloat * get_indexes_from_colored(Imaging::ImageFloat *img, Imaging::ImageFloat *img2=NULL);

  Imaging::ImageFloat *refine_colored(Imaging::ImageFloat *img, Basics::MatrixFloat *mat, int num_classes = 3);
  struct interest_point:AprilUtils::Point<int> {
    bool natural_type;
    int point_class;
    float log_prob;

    interest_point() {}
    interest_point(int x, int y, int point_class, bool type, float log_prob):
      Point<int>(x,y), natural_type(type), point_class(point_class), log_prob(log_prob) {}
    bool operator< (interest_point &ip)
    {
      return this->log_prob > ip.log_prob;
    }

    float angle(interest_point &ip) {
      float deltaY = ip.y - this->y;
      float deltaX = ip.x - this->x;

      return atan2(deltaY, deltaX);
    } 

  };

  class PointComponent:public vector<interest_point> {
  public:
    PointComponent(int size):vector<interest_point>(size){};
    PointComponent():vector<interest_point>(){};

    double line_least_squares();

    PointComponent *get_points_by_type(const int point_class, \
                                       const float min_prob = -999999.00);
    void sort_by_confidence();
    void sort_by_x();
    AprilUtils::line *get_regression_line();
  };
  class SetPoints: public Referenced {
  protected:
    vector< PointComponent > *ccPoints;
    int size;
    int num_points;
    Imaging::ImageFloat *img;

  public:
    SetPoints(Imaging::ImageFloat *img);
    void addComponent();
    void addComponent(PointComponent &);
    int getNumPoints() { return num_points;};
    int getSize() { return size;}
    PointComponent &getComponent(int cc) {
      return (*ccPoints)[cc];
    }
    void addPoint(int component, interest_point ip);
    void addPoint(int component, int x, int y, int c, bool natural_type, float log_prob = 0.0) {
      addPoint(component, interest_point(x,y,c,natural_type,log_prob));
    }
    void print_components();
    void sort_by_confidence();
    void sort_by_x();
    const vector <PointComponent > *getComponents() {
      return ccPoints;
    }
    ~SetPoints(){
      delete ccPoints;
    };
    float component_affinity(int component, interest_point &ip);
    float similarity(interest_point &ip1, interest_point &ip2);

  };

  class ConnectedPoints: public SetPoints {

  private:
    Imaging::ImageConnectedComponents *imgCCs;
  public:
    ConnectedPoints(Imaging::ImageFloat *img);
    void addPoint(interest_point ip);
    void addPoint(int x, int y, int c, bool natural_type, float log_prob = 0.0) {
      addPoint(interest_point(x,y,c,natural_type,log_prob));
    };


    SetPoints *computePoints();
    ~ConnectedPoints() {
      delete imgCCs;
    };
  };

}
#endif
