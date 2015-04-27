#include "gtest.h"
#include "matrixFloat.h"
#include "matrix_ext_blas.h"
#include "matrix_ext_misc.h"
#include "matrix_ext_reductions.h"
#include "smart_ptr.h"

using namespace Basics;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;

MatrixFloat *floatAxpyOperator(MatrixFloat *X, const MatrixFloat *Y) {
  return matAxpy(X, 1.0f, Y);
}

namespace test_matrix {
  
  void check_int(int i)
  {
    EXPECT_TRUE( 0 <= i && i <= 9 );
  }

  TEST(MatrixTest, Test1) {
    SharedPtr<MatrixFloat> m1 = new MatrixFloat(10,20);
    SharedPtr<MatrixFloat> m2 = new MatrixFloat(10,1);
    SharedPtr<MatrixFloat> m3 = new MatrixFloat(10,20);
    
    for (int i=0; i<m1->size(); ++i) {
      (*m1)[i] = static_cast<float>(i);
    }
    for (int i=0; i<m2->size(); ++i) {
      (*m2)[i] = static_cast<float>(i);
    }
    for (int j=0,k=0; j<m1->getDimSize(0); ++j) {
      for (int i=0; i<m1->getDimSize(1); ++i,++k) {
        (*m3)[k] = static_cast<float>(k+j);
      }
    }

    SharedPtr<MatrixFloat> out;
    out = matBroadcast(floatAxpyOperator, m1.get(), m2.get());
    
    EXPECT_TRUE( matEquals(m3.get(), out.get(), 0.01) );
  }
  
}

APRILANN_GTEST_MAIN(test_util)
