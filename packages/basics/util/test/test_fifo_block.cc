#include <iostream>
#include "fifo_block.h"
#include "gtest.h"
using namespace std;

#define N 20
#define EXTRACTN 5
#define BACKN 5
#define BSIZE 7u

namespace test_fifo_block {

  TEST(FifoBlockTest, All) {
    typedef april_utils::fifo_block<int,BSIZE> fifo_block_type;
    fifo_block_type fifobint;
    int i,v;
  
    EXPECT_EQ( fifobint.block_size(), BSIZE );
    EXPECT_TRUE( fifobint.begin() == fifobint.end() );
    EXPECT_TRUE( fifobint.is_end(fifobint.begin()) );
    EXPECT_EQ( fifobint.size(), 0u );
    EXPECT_TRUE( fifobint.empty() );

    // add N elements
    for (i=0; i<N; i++) fifobint.put(i);
  
    fifo_block_type una_copia(fifobint);
  
    EXPECT_EQ( fifobint.size(), static_cast<unsigned int>(una_copia.count()) );
  
    // iterator traversal of copy and original
    fifo_block_type::iterator orig_r = fifobint.begin(), r;
    for (r = una_copia.begin();
         r != una_copia.end() && !fifobint.is_end(orig_r);
         ++r, ++orig_r) {
      ASSERT_EQ( *r, *orig_r );
    }
    ASSERT_TRUE( una_copia.is_end(r) );
    ASSERT_TRUE( r == una_copia.end() );
    ASSERT_TRUE( fifobint.is_end(orig_r) );
    ASSERT_TRUE( orig_r == fifobint.end() );
  
    // extract 5 elements from original object
    for (i=0; i<EXTRACTN && fifobint.get(v); i++) {
      ASSERT_EQ( v, i );
    }
  
    ASSERT_EQ( fifobint.count(), una_copia.count() - EXTRACTN );
  
    // insert values from 0 to 4 in both queues
    for (i=0; i<BACKN ; i++) {
      fifobint.put(i);
      una_copia.put(i);
    }
  
    // assignment operator
    una_copia = fifobint;
  
    for (i=EXTRACTN; i<N ; i++) {
      int v1, v2;
      ASSERT_TRUE( fifobint.get(v1) );
      ASSERT_TRUE( una_copia.get(v2) );
      ASSERT_EQ( v1, v2 );
    }

    for (i=0; i<BACKN ; i++) {
      int v1, v2;
      ASSERT_TRUE( fifobint.get(v1) );
      ASSERT_TRUE( una_copia.get(v2) );
      ASSERT_EQ( v1, v2 );
    }
  
    ASSERT_TRUE( fifobint.empty() );
    ASSERT_TRUE( una_copia.empty() );
  }
}

#undef N
#undef EXTRACTN
#undef BACKN
#undef BSIZE
