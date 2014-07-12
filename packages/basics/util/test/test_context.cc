#include <cstdio>
#include "context.h"
#include "gtest.h"

using april_utils::context;
using namespace std;

// test suite= ContextTest, test name= All
TEST(ContextTest, All) {
  int output[] = { 10, 10, 10, 10, 20, 30, 40, 50, 60, 70, 70, 70, 70 };
  int output_pos = 0;
  context<int> c(3,3);
  c.insert(10);
  int *p = new int(20);
  c.insert(p);
  c.insert(30);
  EXPECT_FALSE( c.ready() ); // Not ready yet
  c.insert(40);
  for (int i=-3; i<4; ++i) {
    EXPECT_EQ( c[i], output[i+3+output_pos] );
  }
  c.insert(50); ++output_pos;
  for (int i=-3; i<4; ++i) {
    EXPECT_EQ( c[i], output[i+3+output_pos] );
  }
  c.insert(60); ++output_pos;
  for (int i=-3; i<4; ++i) {
    EXPECT_EQ( c[i], output[i+3+output_pos] );
  }
  c.insert(70); ++output_pos;
  for (int i=-3; i<4; ++i) {
    EXPECT_EQ( c[i], output[i+3+output_pos] );
  }
  c.end_input();
  c.shift();
  while (c.ready()) {
    ++output_pos;
    for (int i=-3; i<4; ++i) {
      EXPECT_EQ( c[i], output[i+3+output_pos] );
    }
    c.shift();
    EXPECT_LT(output_pos, 13);
  }
}
