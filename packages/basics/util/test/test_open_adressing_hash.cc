
#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;

#include "aux_hash_table.h"
#include "gtest.h"
#include "open_addressing_hash.h"
#include "unused_variable.h"
using namespace AprilUtils;

#define NULL_STRING ""
#define hola "hola"
#define hello "hello"
#define adios "adios"
#define ciao "ciao!"
#define bye "bye"
#define x "x"
#define y "y"
#define z "z"
#define uno "uno"
#define dos "dos"
#define tres "tres"
#define cuatro "cuatro"
#define cinco "cinco"
#define U1 "1"
#define U2 "2"
#define U3 "3"
#define U4 "4"
#define U5 "5"

namespace test_open_adressing_hash {

  typedef open_addr_hash<const char *,const char*> hash_test;

  void process_pair(hash_test::value_type p) {
    UNUSED_VARIABLE(p);
  }

  struct X {
    int data;
   
    X() : data(0) { }

    bool operator == (const X& other) const {
      return data == other.data;
    }

  };

  bool predicado(hash_test::value_type p)
  {
    return (!strcmp(p.second, adios));
  }

  TEST(OpenAdressingHashTable, StringStringHashTable) {
    hash_test a_table(NULL_STRING);
  
    //
    EXPECT_EQ( a_table.size(), 0 );
    a_table[hola]  = hello;
    a_table[adios] = bye;
    a_table.insert(adios,ciao);
    a_table[x] = adios;
    a_table[y] = adios;
    a_table[z] = adios;
    EXPECT_STREQ( a_table[adios], ciao );
    EXPECT_EQ( a_table.size(), 5 );
    //
    a_table[uno] = U1;
    a_table[dos] = U2;
    a_table[tres] = U3;
    a_table[cuatro] = U4;
    a_table[cinco] = U5;
    EXPECT_EQ( a_table.size(), 10 );
    //
    int count = 0;
    for (hash_test::const_iterator i=a_table.begin(); i != a_table.end(); ++i) {
      ++count;
      EXPECT_STREQ( (*i).first, i->first );
      EXPECT_STREQ( (*i).second, i->second );
    }
    EXPECT_EQ( count, a_table.size() );
  
    // Redimensionamos la tabla
    a_table.resize(128);
  
    // Y ahora con for_each
    std::for_each(a_table.begin(), a_table.end(), process_pair);
  }


  TEST(OpenAdressingHashTable, HashFcnAndEqualKey) {
    X a,b,zero;
    a.data=3;
    b.data=3;
  
    // Test con tabla con HashFcn y EqualKey genericas" << endl;
    AprilUtils::open_addr_hash<X, int> t( zero );
    t[a] = 4;
    t[b] = 99;

    for (AprilUtils::open_addr_hash<X,int>::iterator i=t.begin();
         i!=t.end(); i++) {
      ASSERT_EQ( i->first.data, 3 );
      ASSERT_EQ( i->second, 99 );
    }
  
    ASSERT_EQ( t.size(), 1 );
  }
}

#undef hola
#undef hello
#undef adios
#undef ciao
#undef bye
#undef x
#undef y
#undef z
#undef uno
#undef dos
#undef tres
#undef cuatro
#undef cinco
#undef U1
#undef U2
#undef U3
#undef U4
#undef U5
#undef NULL_STRING
