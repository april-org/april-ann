#include <cstdio>
#include "gtest.h"
#include "logbase.h"

#define EPSILON 1e-05

namespace test_logbase {

  TEST(LogbaseTest,All) {

    log_double a,b;
    int i=0;
    a = log_double::from_double(0);
    b = log_double::from_double(0.0001);
    EXPECT_EQ( a, log_double::zero() );
    EXPECT_DOUBLE_EQ( b.log(), -9.210340371976182 );
    
    for (i=1; i<=10; i++) {
      a += b;
      EXPECT_NEAR( b.to_double() * i, a.to_double(), EPSILON );
    }
    
    log_float x  = log_float::from_float(3.0f);
    log_double y = log_double::from_double(5.0);
    log_double z = x+y;
    EXPECT_NEAR( z.log(), 2.0794415416798357, EPSILON );
    EXPECT_NEAR( z.to_double(), 8.0, EPSILON );
    
    z=x*y;
    EXPECT_NEAR( z.log(), 2.70805020110221, EPSILON );
    EXPECT_NEAR( z.to_double(), 15.0, EPSILON );
    
    z=x/y;
    EXPECT_NEAR( z.log(), -0.5108256237659905, EPSILON );
    EXPECT_NEAR( z.to_double(), 0.6, EPSILON );
  }

}

#undef EPSILON
