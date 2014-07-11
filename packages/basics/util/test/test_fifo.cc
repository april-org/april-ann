#include <iostream>
using namespace std;
#include "fifo.h"
#include "gtest.h"

TEST(FifoTest, All) {
  april_utils::fifo<int> f;
  EXPECT_TRUE( f.empty() );
  for (int i=10; i>=0; --i) {
    f.put(i);
    EXPECT_EQ( f.size(), 10 - i + 1 );
  }
  EXPECT_FALSE( f.empty() );
  f.drop_by_value(4);
  EXPECT_EQ( f.size(), 10 );
  int j;
  for (int i=10; i>4; --i) {
    EXPECT_TRUE( f.get(j) );
    EXPECT_EQ( j, i );
  }
  for (int i=3; i>=0; --i) {
    EXPECT_TRUE( f.get(j) );
    EXPECT_EQ( j, i );
  }
  EXPECT_FALSE( f.get(j) );
  EXPECT_TRUE( f.empty() );
  EXPECT_EQ( f.size(), 0 );
}

#undef COMMANDS
