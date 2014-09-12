#include "gtest.h"
#include "smart_ptr.h"

using std::cout;
using std::endl;
using namespace AprilUtils;

namespace smart_ptr_test {

  class A : public Referenced {
  public:
    A() : Referenced() { ++ctor_count; }
    virtual ~A() { ++dtor_count; }
    static int ctor_count;
    static int dtor_count;
  };
  int A::ctor_count = 0;
  int A::dtor_count = 0;

  class B : public A {
  public:
    B() : A() { ++ctor_count; }
    virtual ~B() { ++dtor_count; }
    static int ctor_count;
    static int dtor_count;
  };
  int B::ctor_count = 0;
  int B::dtor_count = 0;

  static void resetCounts() {
    A::ctor_count = 0;
    A::dtor_count = 0;
    B::ctor_count = 0;
    B::dtor_count = 0;
  }

  ////////////////////////////////////////////////////////////////////////////
  
  template<typename Ptr, typename T>
  static Ptr source() {
    return Ptr(new T);
  }
  
  template<typename Ptr>
  static void sink(Ptr) {
  }

  ////////////////////////////////////////////////////////////////////////////

  /*
    TEST(SmartPtrTest, UniquePtr) {
    resetCounts();
    sink(source<UniquePtr<A>, A>());
    sink(source<UniquePtr<B>, B>());
    EXPECT_EQ( A::ctor_count, 2 );
    EXPECT_EQ( A::dtor_count, 2 );
    EXPECT_EQ( B::ctor_count, 1 );
    EXPECT_EQ( B::dtor_count, 1 );
    }
  */

  TEST(SmartPtrTest, SharedPtr) {
    resetCounts();
    sink(source<SharedPtr<A>, A>());
    sink(source<SharedPtr<B>, B>());
    EXPECT_EQ( A::ctor_count, 2 );
    EXPECT_EQ( A::dtor_count, 2 );
    EXPECT_EQ( B::ctor_count, 1 );
    EXPECT_EQ( B::dtor_count, 1 );
  }

  TEST(SmartPtrTest, WeakPtr) {
    resetCounts();
    sink(WeakPtr<A>( SharedPtr<A>(new A)) );
    sink(WeakPtr<B>( SharedPtr<B>(new B)) );
    EXPECT_EQ( A::ctor_count, 2 );
    EXPECT_EQ( A::dtor_count, 2 );
    EXPECT_EQ( B::ctor_count, 1 );
    EXPECT_EQ( B::dtor_count, 1 );
  }
}

#undef N
