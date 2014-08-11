#include "gtest.h"
#include "maxiphobic_minheap.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using std::rand;
using std::srand;

using namespace april_utils;

#define NUM_QUEUES 100000
#define NUM_ELEMENTS 10000000

namespace test_maxiphobic_minheap {

  TEST(MaxiphobicMinheapTest,All) {
    maxiphobic_minheap<int> h[NUM_QUEUES];
    srand(time(NULL));
    for (int i=0; i<NUM_ELEMENTS; i++) {
      int cola = rand()%NUM_QUEUES; 
      if (rand()%2 || h[cola].empty()) {
        h[cola].insert(rand());
      }
      else {
        h[cola].extract_min();
      }
    }
    
    for (int i=0; i<NUM_QUEUES; i++) {
      int min = -1;
      while (!h[i].empty()) {
        int next_min = h[i].extract_min();
        EXPECT_GE( next_min, min );
        min = next_min;
      }
    }

  }

}

#undef NUM_QUEUES
#undef NUM_ELEMENTS

