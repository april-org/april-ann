#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "geometry.h"
namespace april_utils {

    const float infinity = std::numeric_limits<float>::infinity();
    line::line(const Point2D &x, const Point2D &y) {
      float x1 = x.first;
      float y1 = x.second;
      float x2 = y.first;
      float y2 = y.second;
      
      if (x2 == x1) {
        fprintf(stderr, "Warning the slope is infinite");
        m = infinity;
      }
      else {
        m = (y2 - y1)/(x2-x1);
      }
      b = y1 -m*x1;
      
    }
    // static line fromPolar(float phi, float r);
    void line::getPolars(float &phi, float &r) {
        //TODO
        return;

    };

    Point2D line::intersection(const line &out, bool &intersect) {
        // Parallel form
        if (m == out.m){
            intersect = false;
            return (Point2D(infinity, infinity));
        }

        //FIXME: Infinite slope
        float m1 = m;
        float m2 = out.m;
        float b1 = b;
        float b2 = out.b;
        float x = (b2 - b1)/(m1-m2);
        float y = m1*x+b1;

        return Point2D(x,y);
    }

    float line::distance(const Point2D &p) {
      return abs(p.second-m*p.first-b)/sqrt(m*m+1);;
    }
    Point2D line::closestPoint(const Point2D &p, float &dist) {
      int x1 = p.first;
      int y1 = p.second;
      // Extracted from: http://math.ucsd.edu/~wgarner/math4c/derivations/distance/distptline.htm
      float x = (m*y1+x1-m*b)/(m*m+1);
      float y = (m*m*y1+m*x1+b)/(m*m+1);
      dist = distance(p);
      return Point2D(x, y);
    }


}
