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
#include "interest_points.h"
#include "utilImageFloat.h"
#include "utilMatrixFloat.h"
#include "vector.h"
#include "pair.h"
#include "swap.h"
#include "max_min_finder.h"     // para buscar_extremos_trazo
#include "qsort.h"
#include <cmath>
#include <cstdio>

using april_utils::vector;
using april_utils::pair;
using april_utils::min;
using april_utils::max;
using april_utils::max_finder;
using april_utils::min_finder;
using april_utils::swap;
using InterestPoints::Point2D;

namespace InterestPoints {

  struct xy
  {                             // un punto que se compara por la y
    int
      x,
      y;
  xy (int x = 0, int y = 0):x (x), y (y) {
    }
    bool
    operator== (const xy & other) const
    {
      return
        y == other.y;
    }
    bool
    operator< (const xy & other) const
    {
      return
        y <
        other.
        y;
    }
  };

  inline void
  process_stroke_max (max_finder < xy > &finder, vector < xy > *strokep)
  {
    vector < xy > &stroke = *strokep; // mas comodo
    int
      sz = stroke.size ();
    if (sz > 2) {
      for (int i = 0; i < sz; ++i)
        finder.put (stroke[i]);
      finder.end_sequence ();
    }
    stroke.clear ();
  }

  inline void
  process_stroke_min (min_finder < xy > &finder, vector < xy > *strokep)
  {
    vector < xy > &stroke = *strokep; // mas comodo
    int
      sz = stroke.size ();
    if (sz > 2) {
      for (int i = 0; i < sz; ++i)
        finder.put (stroke[i]);
      finder.end_sequence ();
    }
    stroke.clear ();
  }

  inline void
  process_neighborns (vector < Point2D > &v, int line_ini[], int line_end[],
                      int y)
  {

    int
      new_x = line_ini[y] + (line_end[y] - line_ini[y]) / 2;
    //          printf("Adding %d %d\n",new_x, y);
    v.push_back (Point2D (new_x, y));
    line_ini[y] = line_end[y] = -2;

  }
  // If several points have the same y, only keep the central one within
  // a stroke, i.e a straight_line
  inline void
  remove_neighborns (vector < Point2D > &v, int dup_int, int h)
  {
    int
      x_act = 0;
    int *
      line_ini = new int[h];
    int *
      line_end = new int[h];
    vector < Point2D > nV;

    for (int i = 0; i < h; ++i)
      line_ini[i] = line_end[i] = -2;
    int
      sz = v.size ();
    for (int p = 0; p < sz; ++p) {
      int
        x = v[p].first;
      int
        y = v[p].second;
      //                printf("Point %d %d (%d/%d)\n", x, y, x_act, nV.size());

      if (x != x_act)
        x_act = x;
      //Case 1: new line
      if (line_end[y] != x - 1) {
        // Check if is there any line
        if (line_end[y] != -2) {
          process_neighborns (nV, line_ini, line_end, y);
        }
        line_ini[y] = line_end[y] = x;
      }
      else {
        //Case 2: continue line
        line_end[y] = x;

        if (line_end[y] - line_ini[y] > dup_int) {
          // Process the line
          process_neighborns (nV, line_ini, line_end, y);
        }

      }
    }
    // Check if exists unprocessed_neighborns
    for (int i = 0; i < h; ++i) {
      if (line_end[i] != -2) {
        process_neighborns (nV, line_ini, line_end, i);
      }

    }

    v.swap (nV);

    delete[]line_ini;
    delete[]line_end;
  }
  // Returns true if white, false if not
  // It is assumed that white is 1 and 0
  // otherwise use flag reverse
  inline
    bool
  is_white (float value, float threshold, bool reverse = false) {

    if (value > threshold)
      return true xor reverse;

    if (value == threshold)
      return true;

    return false xor reverse;
  }

  // Returns true if white, false if not
  // It is assumed that white is 1 and 0
  // otherwise use flag reverse
  inline
    bool
  is_black (float value, float threshold, bool reverse = false) {

    if (value < threshold)
      return true xor reverse;

    if (value == threshold)
      return true;

    return false xor reverse;
  }
  april_utils::vector < Point2D >
    *extract_points_from_image_old (ImageFloat * pimg, float threshold_white,
                                    float threshold_black, int local_context,
                                    int duplicate_interval) {

    /*        const int          contexto = 6;
       const float threshold_white = 0.6; // <= es blanco
       const float threshold_black = 0.4; // >= es negro
       const float duplicate_interval = 2;
     */
    ImageFloat & img = *pimg;   // mas comodo
    int
      x,
      y,
      h = img.height, w = img.width;

    int *
      stamp_max = new int[h];
    int *
      stamp_min = new int[h];
    vector < xy > **stroke_vec_max = new vector < xy > *[h];  // resultado
    vector < xy > **stroke_vec_min = new vector < xy > *[h];  // resultado
    for (y = 0; y < h; ++y) {
      stroke_vec_max[y] = new vector < xy >;
      stroke_vec_min[y] = new vector < xy >;
      stamp_max[y] = -1;
      stamp_min[y] = -1;
    }
    vector < xy > result_xy;
    max_finder < xy > maxf (local_context, local_context, &result_xy);
    min_finder < xy > minf (local_context, local_context, &result_xy);

    // avanzamos columna a columna por toda la imagen
    for (x = 0; x < w; ++x) {
        // el borde inferior de los trazos, subiendo en la columna
        for (y = h - 1; y > 0; --y) {
            if ((y == h - 1 || is_white (img (x, y + 1), threshold_white)) && (is_black (img (x, y - 1), threshold_black))) { 
                int
                    index = -1;
                if (stamp_max[y] == x)
                    index = y;
                else if (y - 1 >= 0 && stamp_max[y - 1] == x)
                    index = y - 1;
                else if (y + 1 < h && stamp_max[y + 1] == x)
                    index = y + 1;
                else if (y - 2 >= 0 && stamp_max[y - 2] == x)
                    index = y - 2;
                else if (y + 2 < h && stamp_max[y + 2] == x)
                    index = y + 2;
                else {
                    process_stroke_max (maxf, stroke_vec_max[y]);
                    index = y;
                }
                stroke_vec_max[index]->push_back (xy (x, y));
                if (index != y) {
                    swap (stroke_vec_max[y], stroke_vec_max[index]);
                }
                stamp_max[y] = x + 1;
                //
                --y;
            }
        }

        // el borde superior de los trazos, bajando en la columna
        for (y = 0; y < h - 1; ++y) {
            if (is_black (img (x, y + 1), threshold_black) &&
                    (y == 0 || is_white (img (x, y - 1), threshold_white))) {

                int
                    index = -1;
                if (stamp_min[y] == x)
                    index = y;
                else if (y - 1 >= 0 && stamp_min[y - 1] == x)
                    index = y - 1;
                else if (y + 1 < h && stamp_min[y + 1] == x)
                    index = y + 1;
                else if (y - 2 >= 0 && stamp_min[y - 2] == x)
                    index = y - 2;
                else if (y + 2 < h && stamp_min[y + 2] == x)
                    index = y + 2;
                else {
                    process_stroke_min (minf, stroke_vec_min[y]);
                    index = y;
                }
                stroke_vec_min[index]->push_back (xy (x, y));
                if (index != y) {
                    swap (stroke_vec_min[y], stroke_vec_min[index]);
                }
                stamp_min[y] = x + 1;
                ++y;
            }
        }
    }
    for (y = 0; y < h; ++y) {
        process_stroke_max (maxf, stroke_vec_max[y]);
        process_stroke_min (minf, stroke_vec_min[y]);
        delete stroke_vec_max[y];
        delete stroke_vec_min[y];
    }
    delete[]stroke_vec_max;
    delete[]stroke_vec_min;
    delete[]stamp_max;
    delete[]stamp_min;
    // convertir stroke_set a Point2D
    int
        sz = result_xy.size ();
    vector < Point2D > *result_Point2D = new vector < Point2D > (sz);
    vector < Point2D > &vec = *result_Point2D;
    for (int i = 0; i < sz; ++i) {
        vec[i].first = result_xy[i].x;
        vec[i].second = result_xy[i].y;
    }

    //Delete duplicates
    remove_neighborns (vec, duplicate_interval, h);
    return result_Point2D;

    }

  void extract_points_from_image (ImageFloat * pimg,
          vector < Point2D > *local_minima,
          vector < Point2D > *local_maxima,
          float threshold_white, float threshold_black,
          int local_context, int duplicate_interval)
  {

      /*        const int          contexto = 6;
                const float threshold_white = 0.6; // <= es blanco
                const float threshold_black = 0.4; // >= es negro
                const float duplicate_interval = 2;
                */
      ImageFloat & img = *pimg;   // mas comodo
      int
          x,
          y,
          h = img.height, w = img.width;

      int *
          stamp_max = new int[h];
      int *
          stamp_min = new int[h];
      vector < xy > **stroke_vec_max = new vector < xy > *[h];  // resultado
      vector < xy > **stroke_vec_min = new vector < xy > *[h];  // resultado
      for (y = 0; y < h; ++y) {
          stroke_vec_max[y] = new vector < xy >;
          stroke_vec_min[y] = new vector < xy >;
          stamp_max[y] = -1;
          stamp_min[y] = -1;
      }
      vector < xy > result_max;
      vector < xy > result_min;
      max_finder < xy > maxf (local_context, local_context, &result_max);
      min_finder < xy > minf (local_context, local_context, &result_min);

      // avanzamos columna a columna por toda la imagen
      for (x = 0; x < w; ++x) {
          // el borde inferior de los trazos, subiendo en la columna
          for (y = h - 1; y > 0; --y) {
              if ((y == h - 1 || is_white (img (x, y + 1), threshold_white)) && (is_black (img (x, y - 1), threshold_black))) { // procesar el pixel

                  int
                      index = -1;
                  if (stamp_max[y] == x)
                      index = y;
                  else if (y - 1 >= 0 && stamp_max[y - 1] == x)
                      index = y - 1;
                  else if (y + 1 < h && stamp_max[y + 1] == x)
                      index = y + 1;
                  else if (y - 2 >= 0 && stamp_max[y - 2] == x)
                      index = y - 2;
                  else if (y + 2 < h && stamp_max[y + 2] == x)
                      index = y + 2;
                  else {
                      process_stroke_max (maxf, stroke_vec_max[y]);
                      index = y;
                  }
                  stroke_vec_max[index]->push_back (xy (x, y));
                  if (index != y) {
                      swap (stroke_vec_max[y], stroke_vec_max[index]);
                  }
                  stamp_max[y] = x + 1;
                  //
                  --y;
              }
          }
          // el borde superior de los trazos, bajando en la columna
          for (y = 0; y < h - 1; ++y) {
              if (is_black (img (x, y + 1), threshold_black) &&
                      (y == 0 || is_white (img (x, y - 1), threshold_white))) {

                  int
                      index = -1;
                  if (stamp_min[y] == x)
                      index = y;
                  else if (y - 1 >= 0 && stamp_min[y - 1] == x)
                      index = y - 1;
                  else if (y + 1 < h && stamp_min[y + 1] == x)
                      index = y + 1;
                  else if (y - 2 >= 0 && stamp_min[y - 2] == x)
                      index = y - 2;
                  else if (y + 2 < h && stamp_min[y + 2] == x)
                      index = y + 2;
                  else {
                      process_stroke_min (minf, stroke_vec_min[y]);
                      index = y;
                  }
                  stroke_vec_min[index]->push_back (xy (x, y));
                  if (index != y) {
                      swap (stroke_vec_min[y], stroke_vec_min[index]);
                  }
                  stamp_min[y] = x + 1;
                  ++y;
              }
          }
      }
      for (y = 0; y < h; ++y) {
          process_stroke_max (maxf, stroke_vec_max[y]);
          process_stroke_min (minf, stroke_vec_min[y]);
          delete stroke_vec_max[y];
          delete stroke_vec_min[y];
      }
      delete[]stroke_vec_max;
      delete[]stroke_vec_min;
      delete[]stamp_max;
      delete[]stamp_min;
      // convertir stroke_set a Point2D

      int
          sz = result_min.size ();
      //printf("Local maxima %d\n", sz);

      vector < Point2D > &vec_max = *local_maxima;
      for (int i = 0; i < sz; ++i) {
          local_maxima->push_back (Point2D (result_min[i].x, result_min[i].y));
      }

      //Delete duplicates
      remove_neighborns (vec_max, duplicate_interval, h);
      ///                        return result_Point2D;

      sz = result_max.size ();
      //printf("Local minima %d\n", sz);
      //vector<Point2D> *result_Point2D_max = new vector<Point2D>(sz);
      vector < Point2D > &vec_min = *local_minima;
      for (int i = 0; i < sz; ++i) {
          local_minima->push_back (Point2D (result_max[i].x, result_max[i].y));
          //                            vec_max[i].first  = result_max[i].x;
          //                            vec_max[i].second = result_max[i].y;
      }

      //Delete duplicates
      remove_neighborns (vec_min, duplicate_interval, h);
      ///                        return result_Point2D;

  }


  // Class Interest Points
  ConnectedPoints::ConnectedPoints(ImageFloat *img) {

      // Compute connected components of the image
      imgCCs = new ImageConnectedComponents(img);
      ccPoints = new vector< vector<interest_point> >(imgCCs->size);

      fprintf(stderr, "Hay %d componentes\n", imgCCs->size);
      num_points = 0;
  }

  void ConnectedPoints::addPoint(interest_point ip) {

      int x, y;
      x = ip.x;

          // It is local maxima, add 1 to the y for take the component
          if( ip.natural_type) {
              y = ip.y + 1;
          } 
          else {
              y = ip.y - 1;

          }

      int component = this->imgCCs->getComponent(x,y);

      if (component >= 0) {
          (*ccPoints)[component].push_back(ip);
          ++num_points;
      }
      else {
          fprintf(stderr,"Warning the point of interest (%d,%d,%d, %d) is not in any component (%d, %d)\n", ip.x, ip.y, ip.point_class, ip.natural_type, x, y);
      } 
  }
 
  bool componentComparator(vector<interest_point> &v1, vector<interest_point> &v2) {

    if (v1.size() == 0) {
        return false;
    }
    
    if (v2.size() == 0) {
      return true;
    }

    return v1[0] < v2[0];

  }

  bool interestPointXComparator(interest_point &v1, interest_point &v2) {
    return v1.x < v2.x;
  }

  void ConnectedPoints::sort_by_confidence() {

     if (!ccPoints->size()) return;
     for (int i = 0; i < imgCCs->size; ++i) {

         if ((*ccPoints)[i].size() > 0)
           april_utils::Sort(&((*ccPoints)[i])[0], (int)(*ccPoints)[i].size());

     }

     april_utils::Sort(&((*ccPoints)[0]), (int)ccPoints->size(), componentComparator);
    
  }

  void ConnectedPoints::sort_by_x() {
      // Sort first by confidence and then sort by x each component
      sort_by_confidence();

      for (int i = 0; i < imgCCs->size; ++i) {
        
          if ((*ccPoints)[i].size()) {
            april_utils::Sort(&((*ccPoints)[i])[0], (int)(*ccPoints)[i].size(), interestPointXComparator);
          }
      }
  }
  void ConnectedPoints::print_components() {

      for(int i = 0; i < imgCCs->size; ++i) {
          printf("Component %d\n", i);
          printf("\t size %ld\n", (*ccPoints)[i].size());
          for (vector<interest_point>::iterator it = (*ccPoints)[i].begin(); it != (*ccPoints)[i].end(); ++it) {
              printf("\t%d %d %d (%f)\n", it->x, it->y, it->point_class, it->log_prob);
          }

      }
  }
}
// namespace InterestPoints

