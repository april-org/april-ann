#include <cerrno>
#include <cstring>
#include "file_stream.h"
#include "gtest.h"

#define FILE1 "/tmp/APRIL_IO_TEST1.txt"
#define DATA "some data\n\nin several lines\n\n\n\nto test streams.\n"
#define LINE1 "some data"
#define LINE2 "in several lines"
#define LINE3 "to test streams."

namespace april_io {
  
  TEST(FileStream, ConstructorTests) {
    FileStream *f = 0;
    remove(FILE1);
    
    // Constructor read-only open failure
    AssignRef( f, new FileStream(FILE1, "r") );
    EXPECT_TRUE( f->hasError() );
    EXPECT_FALSE( f->isOpened() );
    
    // Constructor write-only
    AssignRef(f, new FileStream(FILE1, "w"));
    EXPECT_FALSE( f->hasError() );
    EXPECT_TRUE( f->isOpened() );
    f->close();
    EXPECT_FALSE( f->hasError() );
    
    // Constructor from opened FILE
    FILE *file = tmpfile();
    AssignRef(f, new FileStream(file));
    EXPECT_FALSE( f->hasError() );
    EXPECT_TRUE( f->isOpened() );
    f->close();
    EXPECT_FALSE( f->hasError() );
    fclose(file);
    
    // Constructor from opened file descriptor
    char tmp_name[] = "/tmp/aXXXXXX";
    int fd = mkstemp(tmp_name);
    EXPECT_GE( fd, 0 );
    AssignRef(f, new FileStream(fd));
    EXPECT_FALSE( f->hasError() );
    EXPECT_TRUE( f->isOpened() );
    f->close();
    EXPECT_FALSE( f->hasError() );
    DecRef(f);
    close(fd);
    //
    remove(FILE1);
    remove(tmp_name);
  }
  
  TEST(FileStream, ReadAndWrite) {
    const size_t N  = strlen(DATA), REP = 10000;
    const size_t N1 = strlen(LINE1);
    const size_t N2 = strlen(LINE2);
    const size_t N3 = strlen(LINE3);
    char *aux = new char[N+1];
    FileStream *f = 0;
    
    // write of a bunch of data
    AssignRef( f, new FileStream(FILE1, "w") );
    EXPECT_FALSE( f->hasError() );
    EXPECT_TRUE( f->isOpened() );
    EXPECT_TRUE( f->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_EQ( f->put(DATA, N), N );
      EXPECT_FALSE( f->hasError() );
    }
    f->close();
    EXPECT_FALSE( f->hasError() );
    
    // read of previous bunch of data
    AssignRef(f, new FileStream(FILE1, "r"));
    EXPECT_FALSE( f->hasError() );
    EXPECT_TRUE( f->isOpened() );
    EXPECT_TRUE( f->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_FALSE( f->eof() );
      EXPECT_EQ( f->get(aux, N), N );
      EXPECT_FALSE( f->hasError() );
      aux[N] = '\0';
      EXPECT_STREQ( aux, DATA );
    }
    EXPECT_EQ( f->get(aux, 1u), 0u ); // just in case to force EOF read
    EXPECT_TRUE( f->eof() );
    f->close();
    EXPECT_FALSE( f->hasError() );
    
    // read of previous bunch of data by lines
    AssignRef(f, new FileStream(FILE1, "r"));
    EXPECT_FALSE( f->hasError() );
    EXPECT_TRUE( f->isOpened() );
    EXPECT_TRUE( f->good() );
    for (unsigned int i=0; i<REP; ++i) {
      EXPECT_FALSE( f->eof() );
      // LINE 1
      EXPECT_EQ( f->get(aux, N, "\r\n"), N1 );
      EXPECT_FALSE( f->hasError() );
      aux[N1] = '\0';
      EXPECT_STREQ( aux, LINE1 );
      // LINE 2
      EXPECT_EQ( f->get(aux, N, "\r\n"), N2 );
      EXPECT_FALSE( f->hasError() );
      aux[N2] = '\0';
      EXPECT_STREQ( aux, LINE2 );
      // LINE 3
      EXPECT_EQ( f->get(aux, N, "\r\n"), N3 );
      EXPECT_FALSE( f->hasError() );
      aux[N3] = '\0';
      EXPECT_STREQ( aux, LINE3 );
    }
    EXPECT_FALSE( f->eof() );
    EXPECT_EQ( f->get(aux, N), 1u ); // just in case to force EOF read
    EXPECT_TRUE( f->eof() );
    f->close();
    EXPECT_FALSE( f->hasError() );
    DecRef(f);
    //
    delete[] aux;
    remove(FILE1);
  }
}

APRILANN_GTEST_MAIN(test_april_io)
