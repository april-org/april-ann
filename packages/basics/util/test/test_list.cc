#define USE_ITERATOR_TAGS
#include "gtest.h"
#include "list.h"

#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;
using namespace AprilUtils;

#define N 10u

namespace test_list {

  struct checkValue {
    int value;
    checkValue(int v) : value(v) { }
    void operator()(const checkValue &a) {
      EXPECT_EQ( a.value, value );
    }
  };

  struct checkNotValue {
    int value;
    checkNotValue(int v) : value(v) { }
    void operator()(const checkNotValue &a) {
      EXPECT_NE( a.value, value );
    }
  };

  struct checkLessEqualValue {
    int value;
    checkLessEqualValue(int v) : value(v) { }
    void operator()(const checkLessEqualValue &a) {
      EXPECT_LE( a.value, value );
    }
  };

  bool predicado(int x)
  {
    return x>90;
  }

  void check_int(int i)
  {
    EXPECT_TRUE( 0 <= i && i <= 9 );
  }

  TEST(ListTest, Test1) {
    int j;
    list<int> l;
    EXPECT_TRUE( l.empty() );
    EXPECT_EQ( l.size(), 0u );
    
    // push elements
    int data[N] = { 4, 5, 6, 7, 3, 2, 1, 8, 9, 0 };
    for (unsigned int i=0; i<N; ++i) {
      l.push_back(data[i]);
    }
    
    EXPECT_FALSE( l.empty() );
    EXPECT_EQ( l.size(), N );
    EXPECT_EQ( l.front(), data[0] );
    EXPECT_EQ( l.back(), data[N-1] );
    
    // for_each traversal
    std::for_each(l.begin(), l.end(), check_int);
    cout << endl;

    // for_each reverse traversal
    std::for_each(l.rbegin(), l.rend(), check_int);
    
    // iterator traversal
    for (list<int>::iterator it = l.begin(); it != l.end(); ++it) {
      check_int( *it );
    }
    
    for (list<int>::const_iterator it = l.begin(); it != l.end(); ++it) {
      check_int( *it );
    }

    // list with elements [4,8) taken from l
    list<int>::iterator first=l.begin();
    list<int>::iterator last =l.end();

    for (int i=0; i<4; i++) first++;
    for (int i=0; i<2; i++) last--;
    list<int> l4(first,last);
    EXPECT_EQ( l4.size(), 4u );

    j=4;
    for (list<int>::iterator it = l4.begin(); it != l4.end(); ++it) {
      EXPECT_EQ( *it, data[j++] );
    }
    
    // pop
    l.pop_front();
    j=1;
    for (list<int>::iterator it = l.begin(); it != l.end(); ++it) {
      EXPECT_EQ( *it, data[j++] );
    }
    
    EXPECT_EQ( l.size(), N - 1u );
    
    // pop
    l.pop_back();
    j=1;
    for (list<int>::iterator it = l.begin(); it != l.end(); ++it) {
      EXPECT_EQ( *it, data[j++] );
    }
    
    EXPECT_EQ( l.size(), N - 2u );
    EXPECT_FALSE( l.empty() );
    
    // insert l at l4
    l4.insert(++l4.begin(), l.begin(), l.end());
    EXPECT_EQ( l4.size(), 4 + l.size() );
    
    // insert 5 times 999
    l4.insert(--l4.end(), 5, 999);
    EXPECT_EQ( l4.size(), 4 + l.size() + 5 );
    
    l4.insert(l4.end(), -6789);
    EXPECT_EQ( l4.size(), 4 + l.size() + 5 + 1 );
    
    // erase second and third elements
    last = ++l4.begin();
    first = last++;
    ++last;
    l4.erase(first, last);
    EXPECT_EQ( l4.size(), 4 + l.size() + 5 + 1 - 2 );

    // resizing
    l4.resize(10u);
    EXPECT_EQ( l4.size(), 10u );
    EXPECT_FALSE( l4.empty() );
    
    // resize with adding -1 values at end
    l4.resize(12u, -1);
    EXPECT_EQ( l4.size(), 12u );
    EXPECT_EQ( l4.back(), -1 );
    
    // splice
    l4.splice(++l4.begin(), l);
    EXPECT_EQ( l.size(), 0u );
    EXPECT_TRUE( l.empty() );
    EXPECT_EQ( l4.size(), 12u + N - 2u );
    
    // remove
    l4.remove(6);
    std::for_each(l4.begin(), l4.end(), checkNotValue(6));
    
    // remove if x>90
    l4.remove_if(predicado);
    std::for_each(l4.begin(), l4.end(), checkLessEqualValue(90));
    
    // now some STL algorithms ;)
    l4.erase(std::find(l4.begin(), l4.end(), 7));
    
    list<int>::iterator i = std::find(l4.begin(), l4.end(), 5);
    l4.splice(l4.begin(), l4, i);
    
    *l4.begin()=99;
  }
  
  TEST(ListTest, Test2) {
    // 12 elements with value 10
    list<int> l2(12, 10);
    std::for_each(l2.begin(), l2.end(), checkValue(10));
    
    // 10 elements with value -1234
    list<int> l3(10, -1234);
    std::for_each(l3.begin(), l3.end(), checkValue(-1234));
    
    l2.swap(l3);
    std::for_each(l2.begin(), l2.end(), checkValue(-1234));
    std::for_each(l3.begin(), l3.end(), checkValue(10));
    
    //
    l3.clear();
    EXPECT_EQ( l3.size(), 0u );
    EXPECT_TRUE( l3.empty() );
  }

}

#undef N
