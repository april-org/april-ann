#include <algorithm>
#include <numeric>
#include <iostream>
#include "gtest.h"
#include "slist.h"

#define N 5u

using april_utils::slist;
using std::cout;
using std::endl;

namespace slist_test {

  void check_ne_3(int i)
  {
    EXPECT_NE( i, 3 );
  }

  TEST(SlistTest, All) {
    int data[N] = { 1, 2, 3, 4, 99 };
    april_utils::slist<int> l;

    EXPECT_EQ( l.size(), 0u );
    EXPECT_TRUE( l.empty() );

    for (unsigned int i=0; i<N; ++i)
      l.push_back( data[i] );

    EXPECT_EQ( l.size(), N );
    EXPECT_FALSE( l.empty() );
    
    april_utils::slist<int>::iterator it = l.begin();
    april_utils::slist<int>::const_iterator c_it = l.begin();
    
    EXPECT_TRUE( l.begin() != l.end() );
    EXPECT_TRUE( it != l.end() );
    EXPECT_TRUE( c_it != l.end() );
    
    // iterator traversal
    int j = 0;
    for (april_utils::slist<int>::iterator i = l.begin(); i != l.end(); i++,j++)
      EXPECT_EQ( *i, data[j] );
    
    // const_iterator traversal
    j=0;
    for (april_utils::slist<int>::const_iterator i = l.begin();
         i != l.end(); i++,j++)
      EXPECT_EQ( *i, data[j] );

    // replace 3 by 99
    std::replace(l.begin(), l.end(), 3, 99);
    // traversal checking all != 3
    std::for_each(l.begin(), l.end(), check_ne_3);

    // TODO: Un test para slist::transfer_front_to_front()
    // TODO: Un test para slist::transfer_front_to_back()
  }
}

#undef N
