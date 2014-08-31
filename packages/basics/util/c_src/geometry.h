/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez, Salvador España-Boquera, Joan Pastor-Pellicer
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
#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "pair.h"
namespace AprilUtils {

    template <typename T>
    struct Point {
      T x;
      T y;
      Point(T x, T y):x(x),y(y){};
      Point(){};
      Point<T>(const Point<int> &p){
       x = static_cast<T>(p.x);
       y = static_cast<T>(p.y);   
      };
    };
    typedef Point<float> Point2D;

    
    // TODO: Define Point2D
    class line {

        protected:
            // Slope, y-intercept
            float m, b;

        public:
            line(float m, float b): m(m),b(b){};
            
            template <typename T>
            line(const Point<T> &, const Point<T> &);
            // static line fromPolar(float phi, float r);
            float getSlope() { return m; };
            float getYintercept() { return b; };

            //void getPolars(float &phi, float &r);

            Point2D intersection(const line &, bool &intersect);

            template <typename T>
            Point2D closestPoint(const Point<T> &, float &);

            template <typename T>
            float distance(const Point<T> &);
            //static void polarToRect(float , float , float &, float &);
            //static void rectToPolar(float,  float , float &, float &);
    };

    // TODO: Point/Angle Functions
    // Derivate interest_point Point2D
}

#endif
