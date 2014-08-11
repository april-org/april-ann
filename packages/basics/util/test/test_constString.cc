#include <cstring>
#include <iostream>
#include "constString.h"
#include "gtest.h"

using namespace std;

#define N 4

namespace test_conststring {

  TEST(ConstStringTest, All) {
    const char *strings[N] = { "aaaa", "aab", "aa", "aaa" };
  
    for (int i=0; i<N; ++i) {
      constString s1(strings[i]);
      for (int j=0; j<N; ++j) {
        constString s2(strings[j]);
        EXPECT_EQ( s1 == s2, strcmp(strings[i],strings[j]) == 0 );
        EXPECT_EQ( s1 <= s2, strcmp(strings[i],strings[j]) <= 0 );
        EXPECT_EQ( s1 < s2, strcmp(strings[i],strings[j]) < 0 );
        EXPECT_EQ( s1 >= s2, strcmp(strings[i],strings[j]) >= 0 );
        EXPECT_EQ( s1 > s2, strcmp(strings[i],strings[j]) > 0 );
      }
    }
  }
}

#undef N
