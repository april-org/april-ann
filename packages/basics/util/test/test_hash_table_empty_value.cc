#include "aux_hash_table.h"
#include "gtest.h"
#include "hash_table.h"
#include "null_types.h"
using namespace april_utils;

#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;

typedef april_utils::hash<const char *,NullType> hash_test;

#define hola "hola"
#define adios "adios"
#define uno "uno"
#define dos "dos"
#define tres "tres"
#define cuatro "cuatro"
#define cinco "cinco"

namespace test_hash_table_empty_value {

  void check_pair(hash_test::value_type p)
  {
    EXPECT_STRNE( p.first, adios );
  }

  TEST(HashTableEmptyValues, All) {
    hash_test a_table;
    
    EXPECT_EQ( a_table.size(), 0 );
    EXPECT_TRUE( a_table.empty() );
    
    a_table[hola]  = NullType();
    a_table[adios] = NullType();
    a_table.insert(adios,NullType());
    a_table.erase(adios);
    
    EXPECT_EQ( a_table.size(), 1 );
    EXPECT_FALSE( a_table.empty() );
    EXPECT_FALSE( a_table.search(adios) );
    
    // insert 5 new elements
    a_table[uno]    = NullType();
    a_table[dos]    = NullType();
    a_table[tres]   = NullType();
    a_table[cuatro] = NullType();
    a_table[cinco]  = NullType();
    
    EXPECT_EQ( a_table.size(), 6 );
    
    // iterator traversal
    for (hash_test::const_iterator i=a_table.begin(); i != a_table.end(); ++i) {
      EXPECT_EQ( (*i).first, i->first );
      check_pair( *i );
    }
    
    // for_each traversal
    std::for_each(a_table.begin(), a_table.end(), check_pair);
  }
}

#undef hola
#undef adios
#undef uno
#undef dos
#undef tres
#undef cuatro
#undef cinco
