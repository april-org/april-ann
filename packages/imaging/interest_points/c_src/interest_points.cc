#include "interest_points.h"
#include "utilImageFloat.h"
#include "utilMatrixFloat.h"
#include "vector.h"
#include "pair.h"
#include "swap.h"
#include "max_min_finder.h" // para buscar_extremos_trazo
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

    struct xy { // un punto que se compara por la y
        int x,y;
        xy(int x=0, int y=0) : x(x),y(y) {}
        bool operator== (const xy& other) const { return y==other.y; }
        bool operator<  (const xy& other) const { return y <other.y; }
    };

    inline
        void process_stroke_max(max_finder<xy> &finder,
                vector<xy> *strokep) {
            vector<xy> &stroke = *strokep; // mas comodo
            int sz = stroke.size();
            if (sz > 2) {
                for (int i=0;i<sz;++i)
                    finder.put(stroke[i]);
                finder.end_sequence();
            }
            stroke.clear();
        }

    inline
        void process_stroke_min(min_finder<xy> &finder,
                vector<xy> *strokep) {
            vector<xy> &stroke = *strokep; // mas comodo
            int sz = stroke.size();
            if (sz > 2) {
                for (int i=0;i<sz;++i)
                    finder.put(stroke[i]);
                finder.end_sequence();
            }
            stroke.clear();
        }



    vector<Point2D>* extract_points_from_image(ImageFloat *pimg) {

        const int          contexto = 6;
        const float threshold_white = 0.4; // <= es blanco
        const float threshold_black = 0.6; // >= es negro

        ImageFloat &img = *pimg; // mas comodo
        int x,y,h=img.height,w=img.width;

        int *stamp_max = new int[h];
        int *stamp_min = new int[h];
        vector<xy> **stroke_vec_max = new vector<xy>*[h]; // resultado
        vector<xy> **stroke_vec_min = new vector<xy>*[h]; // resultado
        for (y = 0; y < h; ++y) {
            stroke_vec_max[y] = new vector<xy>;
            stroke_vec_min[y] = new vector<xy>;
            stamp_max[y]      = -1;
            stamp_min[y]      = -1;
        }
        vector<xy> result_xy;
        max_finder<xy> maxf(contexto,contexto,&result_xy);
        min_finder<xy> minf(contexto,contexto,&result_xy);

        // avanzamos columna a columna por toda la imagen
        for (x = 0; x < w; ++x) {
            // el borde inferior de los trazos, subiendo en la columna
            for (y = h-1; y > 0; --y) {
                if ((y==h-1 || (img(x,y+1) <= threshold_white)) &&
                        (img(x,y-1) >= threshold_black)) { // procesar el pixel
                    int index=-1;
                    if (stamp_max[y] == x) index=y;
                    else if (y-1>=0 && stamp_max[y-1] == x) index=y-1;
                    else if (y+1<h  && stamp_max[y+1] == x) index=y+1;
                    else if (y-2>=0 && stamp_max[y-2] == x) index=y-2;
                    else if (y+2<h  && stamp_max[y+2] == x) index=y+2;
                    else {
                        process_stroke_max(maxf,stroke_vec_max[y]);
                        index = y;
                    }
                    stroke_vec_max[index]->push_back(xy(x,y));
                    if (index != y) { swap(stroke_vec_max[y],stroke_vec_max[index]); }
                    stamp_max[y] = x+1;
                    //
                    --y;
                }
            }
            // el borde superior de los trazos, bajando en la columna
            for (y = 0; y < h-1; ++y) {
                if ( (img(x,y+1) >= threshold_black) &&
                        (y==0 || img(x,y-1) <= threshold_white) ) {
                    // procesar el pixel
                    int index=-1;
                    if (stamp_min[y] == x) index=y;
                    else if (y-1>=0 && stamp_min[y-1] == x) index=y-1;
                    else if (y+1<h  && stamp_min[y+1] == x) index=y+1;
                    else if (y-2>=0 && stamp_min[y-2] == x) index=y-2;
                    else if (y+2<h  && stamp_min[y+2] == x) index=y+2;
                    else {
                        process_stroke_min(minf,stroke_vec_min[y]);
                        index = y;
                    }
                    stroke_vec_min[index]->push_back(xy(x,y));
                    if (index != y) { swap(stroke_vec_min[y],stroke_vec_min[index]); }
                    stamp_min[y] = x+1;
                    ++y;
                }
            }
        }
        for (y = 0; y<h; ++y) {
            process_stroke_max(maxf,stroke_vec_max[y]);
            process_stroke_min(minf,stroke_vec_min[y]);
            delete stroke_vec_max[y];
            delete stroke_vec_min[y];
        }
        delete[] stroke_vec_max;
        delete[] stroke_vec_min;
        delete[] stamp_max;
        delete[] stamp_min;
        // convertir stroke_set a Point2D
        int sz = result_xy.size();
        vector<Point2D> *result_Point2D = new vector<Point2D>(sz);
        vector<Point2D> &vec = *result_Point2D;
        for (int i=0;i<sz;++i) {
            vec[i].first  = result_xy[i].x;
            vec[i].second = result_xy[i].y;
        }
        return result_Point2D;

    }

} // namespace InterestPoints
