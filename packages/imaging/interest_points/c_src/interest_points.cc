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
#include "linear_least_squares.h"

using namespace april_utils;
/*::vector;
using april_utils::pair;
using april_utils::min;
using april_utils::max;
using april_utils::max_finder;
using april_utils::min_finder;
using april_utils::swap;
using april_utils::Point2D;
*/

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
        x = v[p].x;
      int
        y = v[p].y;
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
  inline bool is_black (float value, float threshold, bool reverse = false) {

    if (value < threshold)
      return true xor reverse;

    if (value == threshold)
      return true;

    return false xor reverse;
  }
  april_utils::vector < Point2D >
    *extract_points_from_image_old (
            ImageFloat * pimg, float threshold_white,
            float threshold_black, int local_context,
            int duplicate_interval) {

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
            vec[i].x = result_xy[i].x;
            vec[i].y = result_xy[i].y;
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
          //                            vec_max[i].x  = result_max[i].x;
          //                            vec_max[i].y = result_max[i].y;
      }

      //Delete duplicates
      remove_neighborns (vec_min, duplicate_interval, h);
      ///                        return result_Point2D;

  }

  static Point2D get_next_point(vector<Point2D> v, int index, int width, float default_y)
  {
      assert(index+1 >= 0 && "Invalid index");

      if (v.size() > 0) {
          if (index < int(v.size())-1) {
              return v[index+1];
          }
          else {
              return Point2D(width-1.0f, v.back().y);
          }
      } else {
          return Point2D(width-1.0f, default_y);
      }
  }

  static Point2D get_first_point(vector<Point2D> v, float default_y, int *index)
  {
      Point2D result;
      if (!v.empty()) {
          result.x  = 0;
          result.y = v[0].y;
          if (v[0].x == 0) {
              *index = 0; // index is the last index we have used in v
          } 
          else {
              *index = -1;
          }
      } 
      else {
          result.x = 0.0f;
          result.y = default_y;
          *index = -1;
      }

      return result;
  }

  static void classify_pixel(ImageFloat *img, ImageFloat *result, int col, int ini_row, int end_row, float value) {

      for(int row = ini_row; row < end_row; ++row) {
          if (is_black((*img)(col,row), 0.5)) {
              (*result)(col,row) = value;
          }

      } 


  }
  const float ASC  = 0.2;
  const float BODY = 0.4;
  const float DESC = 0.6;

  ImageFloat *get_pixel_area(ImageFloat *source,
          vector<Point2D> ascenders,
          vector<Point2D> upper_baseline, 
          vector<Point2D> lower_baseline,
          vector<Point2D> descenders
          ){ 

      assert(!upper_baseline.empty() && "Upper baseline must not be empty");
      assert(!lower_baseline.empty() && "Lower baseline must not be empty");

      int width = source->width;
      int height = source->height;

      /* Definimos 4 alturas relevantes en las imagenes
       *  
       *   IMAGEN DESTINO              IMAGEN ORIGEN
       *
       *   - 0 --------------------------- cur_asc -
       *   |               ascenders               |
       *   - dst_upper ----------------- cur_upper -
       *   |                  body                 |
       *   - dst_lower ----------------- cur_lower -
       *   |               descenders              |
       *   - dst_height-1 --------------- cur_desc -
       */



      int dims[2] = {height, width};
      MatrixFloat *result_mat = new MatrixFloat(2, dims);
      result_mat->fill(1.0);
      ImageFloat  *result = new ImageFloat(result_mat);



      int asc_idx = 0;
      int upper_idx = 0;
      int lower_idx = 0;
      int desc_idx = 0;

      Point2D next_asc, next_upper, next_lower, next_desc;
      Point2D prev_asc, prev_upper, prev_lower, prev_desc;

      prev_asc = get_first_point(ascenders, 0.0f, &asc_idx);
      next_asc = get_next_point(ascenders, asc_idx, width, 0.0f);
      asc_idx++;

      prev_desc = get_first_point(descenders, height-1.0f, &desc_idx);
      next_desc = get_next_point(descenders, desc_idx, width, height-1.0f);
      desc_idx++;

      // Default value won't be used due to precondition
      prev_upper = get_first_point(upper_baseline, -9999.9f, &upper_idx);
      next_upper = get_next_point(upper_baseline, upper_idx, width, -9999.9f);
      upper_idx++;

      // Default value won't be used due to precondition
      prev_lower = get_first_point(lower_baseline, -9999.9f, &lower_idx);
      next_lower = get_next_point(lower_baseline, lower_idx, width, -9999.9f);
      lower_idx++;

      int body_columns=0;
      float body_size_sum=0.0f;

      int BASELINE_SLACK=int(0.02f*height);

      for (int column = 0; column < width; column++) {
          if (column > next_asc.x) {
              prev_asc = next_asc;
              next_asc = get_next_point(ascenders, asc_idx, width, 0.0f);
              asc_idx++;
          }
          if (column > next_desc.x) {
              prev_desc = next_desc;
              next_desc = get_next_point(descenders, desc_idx, width, height-1.0f);
              desc_idx++;
          }
          if (column > next_upper.x) {
              prev_upper = next_upper;
              next_upper = get_next_point(upper_baseline, upper_idx, width, 9999.9f);
              upper_idx++;
          }
          if (column > next_lower.x) {
              prev_lower = next_lower;
              next_lower = get_next_point(lower_baseline, lower_idx, width, 9999.9f);
              lower_idx++;
          }

          float cur_upper = max(0.0f, prev_upper.y + 
                  ((column - prev_upper.x) / (next_upper.x-prev_upper.x)) * 
                  (next_upper.y - prev_upper.y) - BASELINE_SLACK);
          float cur_lower = min(height - 1.0f, prev_lower.y + 
                  ((column - prev_lower.x) / (next_lower.x - prev_lower.x)) *
                  (next_lower.y - prev_lower.y) + BASELINE_SLACK);

          if (cur_upper > cur_lower) {
              swap(cur_upper, cur_lower);
          }


          float cur_asc   = min(cur_upper, prev_asc.y + 
                  ((column - prev_asc.x) / (next_asc.x - prev_asc.x) ) *
                  (next_asc.y   - prev_asc.y));
          float cur_desc  = max(cur_lower, prev_desc.y +
                  ((column - prev_desc.x) / (next_desc.x - prev_desc.x)) *
                  (next_desc.y  - prev_desc.y));

          // Classify the pixels

          int asc = round(cur_asc);
          int upper = round(cur_upper);
          int lower = round(cur_lower);
          int desc = round(cur_desc);
          //          printf("%d %d %d %d %d\n", column, asc, upper, lower, desc); 
          classify_pixel(source, result, column, asc, upper, ASC);
          classify_pixel(source, result, column, upper, lower, BODY);
          classify_pixel(source, result, column, lower, desc, DESC);

      }
      return result;
  }

  MatrixFloat * get_indexes_from_colored(ImageFloat *img) {

      // Compute the number of pixels
      int total = 0;

      int width  = img->width;
      int height = img->height;
                  float eps = 0.05;
      for (int col = 0; col < width; ++col)
          {
              for(int row = 0; row < height; ++row) {
                  if (is_black((*img)(col,row),DESC+eps)) total++;
              }
          }

          int dims[2] = {total,2};
          MatrixFloat * m_pixels = new MatrixFloat(2, dims);

          int current = 0;
          for (int col= 0; col < width; ++col){

              for(int row = 0; row < height; ++row) {
                  float value = (*img)(col,row);
                  float tag = 0;
                  float index = (row)*width+col+1;

                  if (is_black(value,DESC+eps)) {
                      // Get the label
                      if (value <= ASC+eps) {
                          tag = 1;  
                      }
                      else if(value <= BODY+eps) {
                          tag = 2;
                      }
                      else if(value <= DESC+eps) {
                          tag = 3;
                      }
                      (*m_pixels)(current,0) = index;
                      (*m_pixels)(current,1) = tag;
                      current++;
                  }
              }
          }
          return m_pixels;
  }

  // Class Set Points
  SetPoints::SetPoints(ImageFloat *img) {
      // Compute connected components of the image
      this->img = img;
      ccPoints = new vector< PointComponent >();
      size = 0;
      num_points = 0;
  }

  void SetPoints::addPoint(int component, interest_point ip){
      if (component < 0 || component >= size){
          fprintf(stderr, "Warning the component %d does not exist!! (Total components %d)\n", component, size);    
          return;
      }
      (*ccPoints)[component].push_back(ip);
      ++num_points;

  }

  void SetPoints::addComponent() {
      if ((size_t) size != ccPoints->size())
          fprintf(stderr, "Size sincronization error %d %lu\n", size, ccPoints->size());
      april_assert("Size sincronization error" && (size_t)size == ccPoints->size());

      (*ccPoints).push_back(PointComponent());
      ++size;
  } 
  // Class Interest Points
  ConnectedPoints::ConnectedPoints(ImageFloat *img):  SetPoints::SetPoints(img){
      // Compute connected components of the image
      imgCCs = new ImageConnectedComponents(img);
      ccPoints->resize(imgCCs->size);
      size = imgCCs->size;

      fprintf(stderr, "Hay %d componentes\n", imgCCs->size);
      num_points = 0;
  }

  void SetPoints::addComponent(PointComponent &component) {
      if ((size_t) size != ccPoints->size())
          fprintf(stderr, "Size sincronization error %d %lu\n", size, ccPoints->size());
      april_assert("Size sincronization error" && (size_t)size == ccPoints->size());

      (*ccPoints).push_back(component);
      ++size;
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
          SetPoints::addPoint(component,ip);
          ++num_points;
      }
      else {
          fprintf(stderr,"Warning the point of interest (%d,%d,%d, %d) is not in any component (%d, %d)\n", ip.x, ip.y, ip.point_class, ip.natural_type, x, y);
      } 
  }

  /**
   *
   * Points comparatarors
   *
   **/ 
  bool componentComparator(PointComponent &v1, PointComponent &v2) {

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

  void PointComponent::sort_by_confidence() {
      if (size() > 0)
          april_utils::Sort(&(*this)[0], (int)size());

  }

  void PointComponent::sort_by_x() {

      april_utils::Sort(&(*this)[0], size(), interestPointXComparator);
  }
  void SetPoints::sort_by_confidence() {

      if (!size) return;
      for (int i = 0; i < size; ++i) {
          (*ccPoints)[i].sort_by_confidence();
      }
  }

  void SetPoints::sort_by_x() {
      // Sort first by confidence and then sort by x each component
      sort_by_confidence();

      for (int i = 0; i < size; ++i) {
          (*ccPoints)[i].sort_by_x();
      }
  }
  void SetPoints::print_components() {

      for(int i = 0; i < size; ++i) {
          printf("Component %d\n", i);
          printf("\t size %ld\n", (*ccPoints)[i].size());
          for (PointComponent::iterator it = (*ccPoints)[i].begin(); it != (*ccPoints)[i].end(); ++it) {
              printf("\t%d %d %d (%f)\n", it->x, it->y, it->point_class, it->log_prob);
          }
      }
  }

  // Takes
  float SetPoints::component_affinity(int component, interest_point &ip) {
      if (component < 0 || component >= size){
          fprintf(stderr, "Warning (similarity)! The component %d does not exist!! (Total components %d\n", component, size);    
          return 0.0;
      }
      // If the component is empty the affinity is the probability of the point (logscale)

      PointComponent &cc = (*ccPoints)[component];
      float score_max = 0.0;
      if (cc.size() == 0)
          return ip.log_prob;


      for (size_t p = 0; p < cc.size(); ++p){

          score_max = max(score_max, similarity(cc[p], ip));
      }

      return score_max;
  }

  float angle_diff(interest_point &a, interest_point &b) {

      float alpha = a.angle(b);
      return min(fabs(alpha), fabs(2*M_PI-alpha));
  }

  float SetPoints::similarity(interest_point &ip1, interest_point &ip2) {

      float alpha_threshold = M_PI/8;      
      interest_point *a = &ip1;
      interest_point *b = &ip2;

      // Two cases
      // Are the same class
      if (ip1.point_class == ip2.point_class) {
          if (a->x > b->x){
              a = &ip2;
              b = &ip1;
          }
          //float alpha = fmod(a->angle(*b)+2*M_PI + alpha_threshold/2, (2*M_PI));
          float alpha = angle_diff(*a, *b);
          if (alpha < alpha_threshold)
              return 1.0;
      }
      else {
          // They're are different classes
          return 1.0;

      }
      return 0.0;

  }

  /*  SetPoints * ConnectedPoints::computePoints() {

      SetPoints * mySet = new SetPoints(img);
      int cini = -1;
      int cfin = 0;
      int threshold = 1.0;
  // Process each connected component
  for (int cc = 0; cc < size; ++cc) {
  printf("Computing connected component %d/%d\n", cc, size);

  if (!(*ccPoints)[cc].size())
  continue;
  //Add an empty set
  cini = cfin;
  cfin++;
  mySet->addComponent();
  for (size_t p = 0; p < (*ccPoints)[cc].size(); ++p) {
  bool added = false;
  // for (interest_point ip : (*ccPoints)[cc]) {
  for (int current_set = cini; current_set < cfin; ++current_set) {
  float affinity = mySet->component_affinity(current_set, (*ccPoints)[cc][p]);
  if (affinity >= threshold){
  //Add the point
  mySet->addPoint(current_set, (*ccPoints)[cc][p]);
  added = true;
  break;
  } //added                
  } //components

  if (!added) {
  // Check if it is the first point
  if (p) {
  ++cfin;
  mySet->addComponent();
  }
  mySet->addPoint(cfin-1, (*ccPoints)[cc][p]);
  }

  }//Point
  } 
  return mySet;
  }*/

  SetPoints * ConnectedPoints::computePoints() {
      SetPoints * mySet = new SetPoints(img);
      float threshold = 100.0;
      //Process each component
      for (int cc = 0; cc < size; ++cc) {

          PointComponent &component = (*ccPoints)[cc];
          int n_points = component.size();
          if (n_points == 0)
              continue;

          PointComponent *base_line = component.get_points_by_type(BASELINE);
          double sse = base_line->line_least_squares();

          //Compute the regression over the points of the line
          printf("%d %d %f %f\n", cc, (int)base_line->size(),sse, sse/n_points);

          if (sse/n_points < threshold) {
              mySet->addComponent(*base_line);
          }
      }

      return mySet;
  }

  PointComponent *PointComponent::get_points_by_type(const int point_class, const float min_prob) {
      // TODO: Check if cc is on the range
      PointComponent *l = new PointComponent();

      for(size_t i = 0; i < size(); ++i) {
          interest_point v = (*this)[i];
          if (v.point_class == point_class and v.log_prob > min_prob) {
              l->push_back(v);

          }
      }

      return l; 
  }
  line * PointComponent::get_regression_line() {
      //TODO: Move to geometry
      size_t n = this->size();
      if (n <= 1)
          return NULL;

      double *vx = new double[n];
      double *vy = new double[n];


      for (size_t i = 0; i < n; ++i) {
          vx[i] = (*this)[i].x;
          vy[i] = (*this)[i].y;
      }

      double a, b;
      least_squares(vx,vy, n, a, b);

      delete []vx;
      delete []vy;

      return new line(b,a);
  }

  double PointComponent::line_least_squares() {

      size_t n = size();
      if (n <= 1)
          return 0.0;

      line *myLine = this->get_regression_line();

      double sse = 0.0;

      for (size_t i = 0; i < size(); ++i) {
          double dist =  myLine->distance((*this)[i]); 
          sse += dist*dist; 
      }
      delete myLine;
      return sse;
  }

}

// namespace InterestPoints

