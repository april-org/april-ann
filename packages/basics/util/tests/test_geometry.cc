//#include <iostream>
#include <stdio.h>
#include "geometry.h"
using namespace std;
using namespace april_utils;

//using namespace april_utils;
int main() {

    // Given two points (2,2) -> (6,3)
    april_utils::Point2D p1 = april_utils::Point2D(3,2);
    april_utils::Point2D p2 = april_utils::Point2D(6,3);

    //Define the line
    april_utils::line morgan = april_utils::line(p1,p2);
    printf("Line created, Slope: %f, Y-intercept: %f\n", morgan.getSlope(), morgan.getYintercept());
   
    // Create a new line
    april_utils::Point2D p3 = april_utils::Point2D(4, 1);
    april_utils::Point2D p4 = april_utils::Point2D(6, 0.5);
    april_utils::line rect2 = april_utils::line(p3,p4);
    printf("Line created, Slope: %f, Y-intercept: %f\n", rect2.getSlope(), rect2.getYintercept());
    // Check the intersection
    bool flag;
    Point2D pInt = morgan.intersection(rect2, flag);
    printf("Line intersection %f, %f\n", pInt.first, pInt.second);

    //Check the point distance
    Point2D p5 = Point2D(4,4);
    
    float dist = 0.0f;
    Point2D cp = morgan.closestPoint(p5,dist);

    printf("The closest point to (%f,%f) is (%f,%f) at %f units\n", p5.first, p5.second, cp.first, cp.second, dist);
    return 0;
}
