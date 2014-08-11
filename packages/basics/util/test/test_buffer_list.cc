#include "buffer_list.h"
#include "binarizer.h"
#include "gtest.h"

#include <iostream>
using std::cout;
using std::endl;

#define FORMAT_STRING "hola mundo! %d+%d=%d,"
#define BUFFER_CONTENT "!!!$78ASQ88D<f08FrA08H9J+8IXS'8Jz^{8LAgw8Mcps8O)yo8PJ)khola mundo! 0+2=2,hola mundo! 1+2=3,hola mundo! 2+2=4,"

namespace test_buffer_list {

  TEST(BufferListTest, All) {
    buffer_list bl;
  
    for (int i=0;i<3;i++)
      bl.add_formatted_string_right(FORMAT_STRING,i,2,i+2);
    float v[10];
    for (int i=0;i<10;i++) v[i] = 0.81+i*0.1;
    bl.add_binarized_float_left (v, 10);

    uint32_t blsize = bl.get_size();
    // bufferlist size
    EXPECT_EQ( blsize, 104u );
  
    // concat size at start of the buffer (left side)
    bl.add_binarized_uint32_left(&blsize,1);

    char *resul = bl.to_string(buffer_list::NULL_TERMINATED);
  
    // check resul
    EXPECT_STREQ( resul, BUFFER_CONTENT );
  
    delete[] resul;
  }
}

#undef FORMAT_STRING
#undef BUFFER_CONTENT
