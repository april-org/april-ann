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
#include "utilImageFloat.h"
#include "pair.h"
#include "vector.h"
#include "geometry.h"
using april_utils::vector;
using april_utils::Point2D;
namespace InterestPoints
{
  
  enum {ASCENDER, TOPLINE, CENTRAL, BASELINE, DESCENDER, OTHER};
  /**
   * @brief Given an Image returns a vector with the local maxima and local minima of the given image.
   *
   * @param[in] pimg Pointer to the image
   * @param[in] threshold_white More than this value is considered black
   * @param[in] threshold_black Less than this value is set to white
   * @param[in] local_context The number of pixels in stroke that is used to compute the local maxima/minima
   * @param[in] duplicate_interval The minimum distance of locals within the same stroke
   */
  april_utils::vector<Point2D>* extract_points_from_image_old(ImageFloat *pimg, float threshold_white = 0.6, float threshold_black = 0.4, int local_context = 6, int duplicate_interval = 5);

  /**
   * @brief Given an Image returns a vector with the local maxima and local minima of the given image.
   *
   * @param[in] pimg Pointer to the image
   * @param[out] local_minima Return the points of the local minima contour
   * @param[out] local_maxima Return the points of the local maxima contour
   * @param[in] threshold_white More than this value is considered black
   * @param[in] threshold_black Less than this value is set to white
   * @param[in] local_context The number of pixels in stroke that is used to compute the local maxima/minima
   * @param[in] duplicate_interval The minimum distance of locals within the same stroke
   */
  void extract_points_from_image(ImageFloat *pimg, april_utils::vector<Point2D> *local_minima, april_utils::vector<Point2D> *local_maxima, float threshold_white= 0.4, float threshold_black = 0.6, int local_context = 6, int duplicate_interval = 3);

  
 struct interest_point {
   int x;
   int y;
   bool natural_type;
   int point_class;
   float log_prob;
   
   interest_point() {}
   interest_point(int x, int y, int point_class, bool type, float log_prob):
       x(x), y(y), natural_type(type), point_class(point_class), log_prob(log_prob) {}
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

 class SetPoints: public Referenced {
     protected:
         vector< vector<interest_point> > *ccPoints;
         int size;
         int num_points;
         ImageFloat *img;

     public:
         SetPoints(ImageFloat *img);
         void addPoint(int component, interest_point ip);
         void addPoint(int component, int x, int y, int c, bool natural_type, float log_prob = 0.0) {
             addPoint(component, interest_point(x,y,c,natural_type,log_prob));
         };
         void addComponent();
         int getNumPoints() { return num_points;};
         int getSize() { return size;}

         void print_components();
         void sort_by_confidence();
         void sort_by_x();
         const vector <vector <interest_point> > *getComponents() {
           return ccPoints;
         }
         ~SetPoints(){
             delete ccPoints;
         };
         float component_affinity(int component, interest_point &ip);
         float similarity(interest_point &ip1, interest_point &ip2);
         vector<interest_point> *get_points_by_type(const int cc, const int point_class, \
                 const float min_prob = -999999.00);
 };

 class ConnectedPoints: public SetPoints {

     private:
         ImageConnectedComponents *imgCCs;
     public:
         ConnectedPoints(ImageFloat *img);
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
