#include <stdio.h>
#include "geometry.h"
#include "gtest.h"
using namespace std;
using namespace AprilUtils;

namespace test_geometry {
  
  TEST(GeometryTest, All) {
    // Given two points (2,2) -> (6,3)
    AprilUtils::Point2D p1 = AprilUtils::Point2D(3,2);
    AprilUtils::Point2D p2 = AprilUtils::Point2D(6,3);
    
    //Define the line
    AprilUtils::line morgan = AprilUtils::line(p1,p2);
    EXPECT_FLOAT_EQ( morgan.getSlope(), 0.33333334f );
    EXPECT_FLOAT_EQ( morgan.getYintercept(), 1.0f );
    
    // Create a new line
    AprilUtils::Point2D p3 = AprilUtils::Point2D(4, 1);
    AprilUtils::Point2D p4 = AprilUtils::Point2D(6, 0.5);
    AprilUtils::line rect2 = AprilUtils::line(p3,p4);
    EXPECT_FLOAT_EQ( rect2.getSlope(), -0.25f );
    EXPECT_FLOAT_EQ( rect2.getYintercept(), 2.0f );
    
    // Check the intersection
    bool flag;
    Point2D pInt = morgan.intersection(rect2, flag);
    EXPECT_FLOAT_EQ( pInt.x, 1.714286f );
    EXPECT_FLOAT_EQ( pInt.y, 1.571429f );
    
    //Check the point distance
    Point2D p5 = Point2D(4,4);
    
    float dist = 0.0f;
    Point2D cp = morgan.closestPoint(p5,dist);
    EXPECT_FLOAT_EQ( p5.x, 4.0f );
    EXPECT_FLOAT_EQ( p5.y, 4.0f );
    EXPECT_FLOAT_EQ( cp.x, 4.5f );
    EXPECT_FLOAT_EQ( cp.y, 2.5f );
    EXPECT_FLOAT_EQ( dist, 1.581139f );
  }
}
