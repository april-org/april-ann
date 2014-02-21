#include "off_line_text_preprocessing.h"
#include "utilImageFloat.h"
#include "utilMatrixFloat.h"
#include "vector.h"
#include "pair.h"
#include "swap.h"
#include "max_min_finder.h" // para buscar_extremos_trazo
#include "unused_variable.h"
#include <cmath>
#include <cstdio>
#include "interest_points.h"
using april_utils::vector;
using april_utils::pair;
using april_utils::min;
using april_utils::max;
using april_utils::max_finder;
using april_utils::min_finder;
using april_utils::swap;
using april_utils::Point2D;

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

static Point2D get_first_point(vector<Point2D> v, int width, float default_y, int *index)
{
    UNUSED_VARIABLE(width);
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


// Reduce a section of one column to one pixel
static float column_reduce(ImageFloat *src, int col,
        float src_top, float src_bottom) {
    float epsilon = 1e-7;
    assert(src_top >= 0 && "Top source must be >= 0");

    // assert(src_bottom <= src->height ||
    //         (0, printf("Bottom source must be <= src_height. (%f/%d)\n",
    //             src_bottom, src->height)));

    assert(src_top <= src_bottom && "Top pixel has to be lower than bottom");

    int pxl_top = floor(src_top);
    int pxl_bottom = floor(src_bottom);

    float rst_top = src_top - pxl_top;
    float rst_bottom = src_bottom - pxl_bottom;

    float total = 0.0f;
    //if (pxl_top == pxl_bottom)
    //   return (float) (*src)(col, pxl_top);
    for(int x = pxl_top; x < pxl_bottom; ++x) {
        total += (*src)(col, x);
    }
    if (rst_top > epsilon) {
        total -= rst_top * ((*src)(col, pxl_top));
    }
    if (rst_bottom > epsilon) {
        total += rst_bottom * ((*src)(col, pxl_bottom));        
    }

    return total/(src_bottom - src_top);
}

// Given a contour matrix (baseline, topline) and a line points
//
bool xComparator(Point2D &v1, Point2D &v2) {
  return v1.x < v2.x;
}

bool yComparator(Point2D &v1, Point2D &v2) {
  return v1.y < v2.y;
}
bool yComparatorReversed(Point2D &v1, Point2D &v2) {
  return v1.y > v2.y;
}

static void filter_asc(vector<Point2D> *points,
		       MatrixFloat::random_access_iterator &line_it,
		       int width, 
		       float vThreshold = 10.0f,
		       int hThreshold = 20) {
  bool *valid = new bool[width];
  for (int i=0; i<width; ++i) valid[i] = true;
  vector<Point2D> *new_points = new vector<Point2D>();
  
  april_utils::Sort(&(*points)[0],points->size(),yComparator);
  for(int i = 0; i < (int)points->size();++i) {
    Point2D &p = (*points)[i];
    int x = int(p.x);
    if (line_it(x,0) > p.y+vThreshold && valid[x]) {
      new_points->push_back(p);
      int first = max(x-hThreshold,0);
      int last  = min(x+hThreshold,width-1);
      for (int i=first; i<=last; ++i)
	valid[i]=false;
    }
  }
  // sort new_points by x
  april_utils::Sort(&(*new_points)[0],new_points->size(),xComparator); 
  
  // delete points and change to new_points
  vector<Point2D> *aux = points;
  points->swap(*new_points);
  delete new_points;
  delete[] valid;
}

static void filter_desc(vector<Point2D> *points,
			MatrixFloat::random_access_iterator &line_it,
			int width, 
			float vThreshold = 10.0f,
			int hThreshold = 20) {
  bool *valid = new bool[width];
  for (int i=0; i<width; ++i) valid[i] = true;
  vector<Point2D> *new_points = new vector<Point2D>();
  
  april_utils::Sort(&(*points)[0],points->size(),yComparatorReversed);
  for(int i = 0; i < (int)points->size();++i) {
    Point2D &p = (*points)[i];
    int x = int(p.x);
    if (line_it(x,1) < p.y-vThreshold && valid[x]) {
      new_points->push_back(p);
      int first = max(x-hThreshold,0);
      int last  = min(x+hThreshold,width-1);
      for (int i=first; i<=last; ++i)
	valid[i]=false;
    }
  }
  // sort new_points by x
  april_utils::Sort(&(*new_points)[0],new_points->size(),xComparator); 
  
  // delete points and change to new_points
  vector<Point2D> *aux = points;
  points->swap(*new_points);
  delete new_points;
  delete[] valid;
}


// Scales the source column to the target column
// Recieve:
// Two Images (Source and target)
// Column to be reduced
// the initial and final x of the source image
// the initial and final x of the target image
static void resize_index(ImageFloat *src, ImageFloat *dst,
        int col, float src_top, float src_bottom,
        float dst_top, float dst_bottom) {
    //We are assuming that the row of the column can be viewed
    //as a float between range 0 and height (inclusive)
    assert(src_top >= 0 && "Top source must be >= 0");
    assert(dst_top >= 0 && "Top dest must be >= 0");
    assert(src_bottom <= src->height && 
            "Bottom source must be <= src_height");
    assert(dst_bottom <= dst->height && 
            "Bottom dest must be <= dest_height");
    assert(src_top <= src_bottom &&
            "Top src pixel has to be lower than bottom");
    assert(dst_top < dst_bottom &&
            "Top target pixel has to be lower than bottom");

    //if (src_top >= src_bottom) 
    //    src_bottom = src_top+1;
    
    int epsilon      = 1e-7;
    float ratio = (src_bottom - src_top)/(dst_bottom - dst_top);
    float cur_top = src_top;
    float cur_bottom;


    //Compute the first row rest
    int first_row    = (int) ceil(dst_top);
    int last_row     = (int) ceil(dst_bottom);
    float rst_top      = first_row - dst_top;

    if (rst_top > epsilon) {
        cur_bottom = src_top;
        cur_top    = (first_row-dst_top)*ratio;
        (*dst)(col, (int)floor(dst_top)) += column_reduce(src, col, cur_top, cur_bottom);
    }

    for (int row = first_row; row < last_row; ++row)
    {
        // fprintf(stderr, "Assert src_cur %f, src_next: %f, top: %f , bottom: %f, height: %d\n", src_cur, src_cur+ratio, src_top,  src_bottom, src->height);
        cur_bottom =  ((row +1 - dst_top)*ratio) + src_top;
        float dst_value = column_reduce(src, col, cur_top, cur_bottom);
        cur_top = cur_bottom;
        (*dst)(col, row) = dst_value;
    }

    //Compute the last row rest
    float rst_bottom = dst_bottom - last_row;

    if (rst_bottom > epsilon) {
        cur_bottom = src_bottom;
        (*dst)(col, (int)ceil(dst_bottom)) += column_reduce(src, col, cur_top, cur_bottom);
    }

}

namespace OCR {
    namespace OffLineTextPreprocessing {

        ImageFloat *normalize_image(ImageFloat *source, int dst_height) {

            int width = source->width;
            int height = source->height;
            int dims[2] = {dst_height, width};

            MatrixFloat *result_mat = new MatrixFloat(2, dims);
            ImageFloat  *result = new ImageFloat(result_mat);

            for(int col = 0; col < width; ++col) {
                resize_index(source, result, col, 0, height, 0, dst_height);
            }
            return result; 

        }


        ImageFloat *normalize_size (ImageFloat     *source,
                float           ascender_ratio,
                float           descender_ratio,
                vector<Point2D> ascenders,
                vector<Point2D> upper_baseline, 
                vector<Point2D> lower_baseline,
                vector<Point2D> descenders,
                int dst_height,
                bool keep_aspect
                )
        {
            // Precondition: upper_baseline and lower_baseline must contain, at least, one point each
            assert(!upper_baseline.empty() && "Upper baseline must not be empty");
            assert(!lower_baseline.empty() && "Lower baseline must not be empty");

            int width = source->width;
            int height = source->height;
            int BASELINE_SLACK=int(0.02f*height);

            //printf("BASELINE_SLACK = %d\n", BASELINE_SLACK);

            if (dst_height < 0)
                dst_height = source->height;

            int ascender_size  = int(roundf(ascender_ratio*dst_height));
            int descender_size = int(roundf(descender_ratio*dst_height));
            int body_size      = dst_height - ascender_size - descender_size;

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

            int dst_upper = ascender_size;
            int dst_lower = ascender_size + body_size;

            int dims[2] = {dst_height, width};
            MatrixFloat *result_mat = new MatrixFloat(2, dims);
            ImageFloat  *result = new ImageFloat(result_mat);

            int asc_idx = 0;
            int upper_idx = 0;
            int lower_idx = 0;
            int desc_idx = 0;

            Point2D next_asc, next_upper, next_lower, next_desc;
            Point2D prev_asc, prev_upper, prev_lower, prev_desc;

            prev_asc = get_first_point(ascenders, width, 0.0f, &asc_idx);
            next_asc = get_next_point(ascenders, asc_idx, width, 0.0f);
            asc_idx++;

            prev_desc = get_first_point(descenders, width, height-1.0f, &desc_idx);
            next_desc = get_next_point(descenders, desc_idx, width, height-1.0f);
            desc_idx++;

            // Default value won't be used due to precondition
            prev_upper = get_first_point(upper_baseline, width, -9999.9f, &upper_idx);
            next_upper = get_next_point(upper_baseline, upper_idx, width, -9999.9f);
            upper_idx++;

            // Default value won't be used due to precondition
            prev_lower = get_first_point(lower_baseline, width, -9999.9f, &lower_idx);
            next_lower = get_next_point(lower_baseline, lower_idx, width, -9999.9f);
            lower_idx++;

            int body_columns=0;
            float body_size_sum=0.0f;

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

                // Center
                //
                // First normalize the body and calculate the normalization rasize
                float body_ratio = (cur_lower - cur_upper)/body_size;
                if (cur_lower-cur_upper >= 1.0f) {
                    body_columns++;
                    body_size_sum += cur_lower-cur_upper;
                    resize_index(source, result, column, cur_upper, cur_lower, dst_upper, dst_lower);
                }


                // Ascender
                float expected_top = dst_upper*body_ratio;
                float top_cut      = cur_asc;

                if (keep_aspect) {
                    top_cut = cur_upper - expected_top;
                }
                //Fill the blanks

                float dst_asc = 0;

                if (top_cut < 0) {
                    dst_asc = -top_cut/body_ratio;
                    top_cut = 0;
                }
                if(cur_upper - top_cut >= 1.0f)
                  resize_index(source, result, column, top_cut, cur_upper, dst_asc, dst_upper);
                //printf("x=%d, 1=%f, 2=%f, 4=%f, 5=%f\n", column, cur_asc, cur_upper, cur_lower, cur_desc);
                // Descenders
                float expected_bottom =  (dst_height - dst_lower)*body_ratio;
                float dst_desc        =  dst_height;
                float bottom_cut      =  cur_desc;

                if (keep_aspect) {
                    bottom_cut = cur_lower + expected_bottom;
                }
                if (bottom_cut > height) {
                    dst_desc   = dst_lower + (height-cur_lower)/body_ratio; 
                    bottom_cut = height;
                }
                //printf("dst_dsc :%d, dst_height: %d, bottom_cut:%d, height: %d, ratio: %f\n", dst_desc, dst_height, bottom_cut, height, body_ratio); 
                assert(dst_desc <= dst_height && "Something went wrong");
                if( bottom_cut - cur_lower >= 1.0f)
                  resize_index(source, result, column, cur_lower, bottom_cut, dst_lower, dst_desc);
            }

            return result;
        }


        ImageFloat *normalize_size (ImageFloat     *source,
                MatrixFloat *line_mat,
                float           ascender_ratio,
                float           descender_ratio,
                int dst_height,
                bool keep_aspect
                )
        {
            // Precondition: upper_baseline and lower_baseline must contain, at least, one point each
            int width = source->width;
            int height = source->height;

            MatrixFloat::random_access_iterator line_it(line_mat);

            assert(line_mat->getDimSize(0) == width && "The number of columns does not fit");
            assert(line_mat->getDimSize(1) == 4 && "There are no 3 areas on the image");
            if (dst_height < 0)
                dst_height = source->height;

            int ascender_size  = int(roundf(ascender_ratio*dst_height));
            int descender_size = int(roundf(descender_ratio*dst_height));
            int body_size      = dst_height - ascender_size - descender_size;

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

            int dst_upper = ascender_size;
            int dst_lower = ascender_size + body_size;

            int dims[2] = {dst_height, width};
            MatrixFloat *result_mat = new MatrixFloat(2, dims);
            ImageFloat  *result = new ImageFloat(result_mat);
            for (int column = 0; column < width; column++) {

                float cur_upper = line_it(column, 1);
                float cur_lower = line_it(column, 2);

                float cur_asc   = line_it(column, 0);
                float cur_desc  = line_it(column, 3);
                // First normalize the body and calculate the normalization rasize
                float body_ratio = (cur_lower - cur_upper)/body_size;
                if (cur_lower-cur_upper >= 1.0f) {
                    resize_index(source, result, column, cur_upper, cur_lower, dst_upper, dst_lower);
                }


                // Ascender
                float expected_top = dst_upper*body_ratio;
                float top_cut      = cur_asc;

                if (keep_aspect) {
                    top_cut = cur_upper - expected_top;
                }
                //Fill the blanks

                float dst_asc = 0;

                if (top_cut < 0) {
                    dst_asc = -top_cut/body_ratio;
                    top_cut = 0;
                }
                if (cur_upper - top_cut >= 1)
                    resize_index(source, result, column, top_cut, cur_upper, dst_asc, dst_upper);
                //printf("x=%d, 1=%f, 2=%f, 4=%f, 5=%f\n", column, cur_asc, cur_upper, cur_lower, cur_desc);
                // Descenders
                float expected_bottom =  (dst_height - dst_lower)*body_ratio;
                float dst_desc        =  dst_height;
                float bottom_cut      =  cur_desc;

                if (keep_aspect) {
                    bottom_cut = cur_lower + expected_bottom;
                }
                if (bottom_cut > height) {
                    dst_desc   = dst_lower + (height-cur_lower)/body_ratio; 
                    bottom_cut = height;
                }
                assert(dst_desc <= dst_height && "Something went wrong");
                if(bottom_cut - cur_lower >= 1.0f)
                 resize_index(source, result, column, cur_lower, bottom_cut, dst_lower, dst_desc);

            }


            return result;
        }

        // From the points of the topline and baseline, adds the ascenderes and descenders
        MatrixFloat *add_asc_desc (ImageFloat     *img,
                MatrixFloat *line_mat
                )
        {
            // Precondition mat size must be columns
            assert(line_mat->getDimSize(0) == img->width && "Matrix points does not fit with the image");
            int width = img->width;
            int height = img->height;


            // Generate the 4 matrix dim
            int dims[2] = {width, 4};
            MatrixFloat *result = new MatrixFloat(2, dims);
            MatrixFloat::random_access_iterator line_it(line_mat);  
            MatrixFloat::random_access_iterator result_it(result);  
            // Compute local maxima and local minima
            vector<Point2D> *ascenders = new vector<Point2D>();
            vector<Point2D> *descenders = new vector<Point2D>();
            InterestPoints::extract_points_from_image(img, ascenders, descenders, 0.6, 0.4, 6, 15 );


            // Filter the points that are over the size
            filter_asc(ascenders, line_it, width);
            filter_desc(descenders, line_it, width);
            // Compute the interpolated lines
            Point2D next_asc, next_desc;
            Point2D prev_asc, prev_desc;
            int asc_idx = 0;
            int desc_idx = 0;


            prev_asc = get_first_point(*ascenders, width, 0.0f, &asc_idx);
            next_asc = get_next_point(*ascenders, asc_idx, width, 0.0f);
            asc_idx++;

            prev_desc = get_first_point(*descenders, width, height-1.0f, &desc_idx);
            next_desc = get_next_point(*descenders, desc_idx, width, height-1.0f);
            desc_idx++;

            for (int column = 0; column < width; column++) {

                float cur_upper = line_it(column,0);
                float cur_lower = line_it(column,1);
                if (cur_upper > cur_lower) 
                    cur_upper = cur_lower; 
                if (column > next_asc.x) {
                    prev_asc = next_asc;
                    next_asc = get_next_point(*ascenders, asc_idx, width, 0.0f);
                    asc_idx++;
                }
                if (column > next_desc.x) {
                    prev_desc = next_desc;
                    next_desc = get_next_point(*descenders, desc_idx, width, height-1.0f);
                    desc_idx++;
                }

                float cur_asc   = min(cur_upper-1.0f, prev_asc.y + 
                        ((column - prev_asc.x) / (next_asc.x - prev_asc.x) ) *
                        (next_asc.y   - prev_asc.y));
                cur_asc = max(0.0f,cur_asc);
                float cur_desc  = max(cur_lower+1.0f, prev_desc.y +
                        ((column - prev_desc.x) / (next_desc.x - prev_desc.x)) *
                        (next_desc.y  - prev_desc.y));
                cur_desc = min(height-1.0f, cur_desc);
                // Add the new lines and copy the old ones

                if (cur_upper >= cur_lower) {
                    cur_upper = max(cur_lower-1.f, 0.0f);
                }
                //printf("Liada %d %f %f,(%f,%f) (%f,%f)\n", column, cur_asc, cur_upper, prev_asc.x, prev_asc.y, next_asc.x, next_asc.y);
                //printf("Liada2 %d %f %f,(%f,%f) (%f,%f)\n", column, cur_desc, cur_lower, prev_desc.x, prev_desc.y, next_desc.x, next_desc.y);
                result_it(column, 0) = cur_asc;
                result_it(column, 3) = cur_desc;
                result_it(column, 1) = cur_upper;
                result_it(column, 2) = cur_lower; 
            }

            delete ascenders;
            delete descenders;
            return result;
        }

    } //namespace OffLineTextPreprocessing
} // namespace OCR 
