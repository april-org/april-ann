#include <cstdio>
#include <cstdlib>
#include "gtest.h"
#include "vector.h"
#include "max_min_finder.h"

using namespace april_utils;

#define N 14
#define MAXN 2u

namespace test_max_min_finder {

  TEST(MaxMinFinderTest,All) {
    april_utils::vector<int> resultado;
    const int ci = 6;
    const int cd = 6;
    bool findmax = true;
    bool findmin = false;
    max_min_finder<int> finder(ci,cd,findmax,&resultado,findmin,0);
    int data[N] = { -2, 4, 6, 10, 4, 2, 1, 0, 3, 4, 6, 3, 2, 0 };
    for (int i=0; i<N; ++i) {
      finder.put(data[i]);
    }
    finder.end_sequence();
    EXPECT_EQ( resultado.size(), MAXN );
    int expected[MAXN] = { 10, 6 }, j=0;
    for (april_utils::vector<int>::iterator r = resultado.begin(); 
         r != resultado.end(); ++r, ++j) {
      EXPECT_EQ( *r, expected[j] );
    }
  }

}

#undef N
#undef MAXN
