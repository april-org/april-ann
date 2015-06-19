#include <cerrno>
#include <cstring>
#include "file_stream.h"
#include "c_string.h"
#include "gtest.h"
#include "smart_ptr.h"

const char *FILE1 = "/tmp/APRIL_IO_TEST1.txt";
const char *DATA  = "some data\n\nin several lines\n\n\n\nto test streams.\n";
const char *LINE1 = "some data";
const char *LINE2 = "in several lines";
const char *LINE3 = "to test streams.";

const size_t N  = strlen(DATA);
const size_t N1 = strlen(LINE1);
const size_t N2 = strlen(LINE2);
const size_t N3 = strlen(LINE3);

const size_t REP = 10000;

namespace AprilIO {
  
  TEST(FileStream, ConstructorTests) {
    AprilUtils::SharedPtr<StreamInterface> ptr;

    remove(FILE1);
    
    // Constructor read-only open failure
    ptr.reset(new FileStream(FILE1, "r"));
    EXPECT_TRUE( ptr->hasError() );
    EXPECT_FALSE( ptr->isOpened() );
    
    // Constructor write-only
    ptr.reset(new FileStream(FILE1, "w"));
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    
    // Constructor from opened FILE
    FILE *file = tmpfile();
    ptr.reset(new FileStream(file));
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    fclose(file);
    
    // Constructor from opened file descriptor
    char tmp_name[] = "/tmp/aXXXXXX";
    int fd = mkstemp(tmp_name);
    EXPECT_GE( fd, 0 );
    ptr.reset(new FileStream(fd));
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    close(fd);
    //
    remove(FILE1);
    remove(tmp_name);
  }
  
  TEST(FileStream, ReadAndWrite) {
    AprilUtils::UniquePtr<char []> aux( new char[N+1] );
    AprilUtils::SharedPtr<StreamInterface> ptr;
    
    // write of a bunch of data
    ptr.reset( new FileStream(FILE1, "w") );
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    EXPECT_TRUE( ptr->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_EQ( ptr->put(DATA, N), N );
      EXPECT_FALSE( ptr->hasError() );
    }
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    
    // read of previous bunch of data
    ptr.reset( new FileStream(FILE1, "r"));
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    EXPECT_TRUE( ptr->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_FALSE( ptr->eof() );
      EXPECT_EQ( ptr->get(aux.get(), N), N );
      EXPECT_FALSE( ptr->hasError() );
      aux[N] = '\0';
      EXPECT_STREQ( aux.get(), DATA );
    }
    EXPECT_EQ( ptr->get(aux.get(), 1u), 0u ); // just in case to force EOF read
    EXPECT_TRUE( ptr->eof() );
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    
    // read of previous bunch of data by lines
    ptr.reset( new FileStream(FILE1, "r"));
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    EXPECT_TRUE( ptr->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_FALSE( ptr->eof() );
      // LINE 1
      EXPECT_EQ( ptr->get(aux.get(), N, "\r\n"), N1 );
      EXPECT_FALSE( ptr->hasError() );
      aux[N1] = '\0';
      EXPECT_STREQ( aux.get(), LINE1 );
      ptr->get(aux.get(), N, "\r\n"); // blankline
      // LINE 2
      EXPECT_EQ( ptr->get(aux.get(), N, "\r\n"), N2 );
      EXPECT_FALSE( ptr->hasError() );
      aux[N2] = '\0';
      EXPECT_STREQ( aux.get(), LINE2 );
      ptr->get(aux.get(), N, "\r\n"); // blankline
      ptr->get(aux.get(), N, "\r\n"); // blankline
      ptr->get(aux.get(), N, "\r\n"); // blankline
      // LINE 3
      EXPECT_EQ( ptr->get(aux.get(), N, "\r\n"), N3 );
      EXPECT_FALSE( ptr->hasError() );
      aux[N3] = '\0';
      EXPECT_STREQ( aux.get(), LINE3 );
    }
    EXPECT_FALSE( ptr->eof() );
    EXPECT_EQ( ptr->get(aux.get(), N), 1u ); // just in case to force EOF read
    EXPECT_TRUE( ptr->eof() );
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    
    // read into a c_string
    AprilUtils::SharedPtr<CStringStream> c_str;
    c_str.reset( new CStringStream() );
    EXPECT_TRUE( c_str->empty() );
    // EXPECT_TRUE( c_str->good() );
    EXPECT_EQ( c_str->size(), 0u );
    EXPECT_FALSE( c_str->hasError() );
    ptr.reset( new FileStream(FILE1, "r"));
    EXPECT_FALSE( ptr->hasError() );
    EXPECT_TRUE( ptr->isOpened() );
    EXPECT_TRUE( ptr->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_FALSE( ptr->eof() );
      // LINE 1
      EXPECT_EQ( ptr->get(c_str.get(), "\r\n"), N1 );
      EXPECT_FALSE( ptr->hasError() );
      c_str->flush();
      EXPECT_EQ( c_str->get(aux.get(), N1), N1 );
      aux[N1] = '\0';
      EXPECT_STREQ( aux.get(), LINE1 );
      ptr->get(aux.get(), N, "\r\n"); // blankline
      // LINE 2
      EXPECT_EQ( ptr->get(c_str.get(), "\r\n"), N2 );
      EXPECT_FALSE( ptr->hasError() );
      c_str->flush();
      EXPECT_EQ( c_str->get(aux.get(), N2), N2 );
      aux[N2] = '\0';
      EXPECT_STREQ( aux.get(), LINE2 );
      ptr->get(aux.get(), N, "\r\n"); // blankline
      ptr->get(aux.get(), N, "\r\n"); // blankline
      ptr->get(aux.get(), N, "\r\n"); // blankline
      // LINE 3
      EXPECT_EQ( ptr->get(c_str.get(), "\r\n"), N3 );
      EXPECT_FALSE( ptr->hasError() );
      c_str->flush();
      EXPECT_EQ( c_str->get(aux.get(), N3), N3 );
      aux[N3] = '\0';
      EXPECT_STREQ( aux.get(), LINE3 );
      // c_string asserts
      EXPECT_FALSE( c_str->hasError() );
      EXPECT_EQ( c_str->size(), (N1+N2+N3)*(i+1) );
    }
    EXPECT_FALSE( ptr->eof() );
    EXPECT_EQ( ptr->get(aux.get(), N), 1u ); // just in case to force EOF read
    EXPECT_TRUE( ptr->eof() );
    ptr->close();
    EXPECT_FALSE( ptr->hasError() );
    c_str->close();
    //
    remove(FILE1);
  }
}

APRILANN_GTEST_MAIN(test_april_io)
