#include <gtest/gtest.h>
#include "lumin.hpp"

// CPU-only tests - these use the default CPU backend
class CPUMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Ensure we're using CPU backend
    auto backend = lumin::create_cpu_backend();
    lumin::set_default_backend(backend);
  }
};

TEST_F(CPUMatrixTest, CreateAndFill) {
  lumin::Matrix A(3,3);
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      A.data()[i * A.cols() + j] = i + j;
    }
  }

  EXPECT_EQ(A.data()[0], 0);
  EXPECT_EQ(A.data()[2*3 + 1], 3); // row 2, col 1
}

TEST_F(CPUMatrixTest, AddMatrices) {
  lumin::Matrix A(2,2), B(2,2);

  A.data()[0] = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  B.data()[0] = 5; B.data()[1] = 6;
  B.data()[2] = 7; B.data()[3] = 8;

  lumin::Matrix C = A.add(B);

  EXPECT_EQ(C.data()[0], 6);
  EXPECT_EQ(C.data()[3], 12);
}

TEST_F(CPUMatrixTest, MultiplyMatrices) {
  lumin::Matrix A(2, 3), B(3, 1);
  A.data()[0] = 1; A.data()[1] = 2; A.data()[2] = 3;
  A.data()[3] = 4; A.data()[4] = 5; A.data()[5] = 6;

  B.data()[0] = 7;
  B.data()[1] = 8;
  B.data()[2] = 9;

  lumin::Matrix C = A.multiply(B);

  EXPECT_EQ(C.rows(), 2);
  EXPECT_EQ(C.cols(), 1);

  EXPECT_EQ(C.data()[0], 50);
  EXPECT_EQ(C.data()[1], 122);
}

TEST_F(CPUMatrixTest, MultiplyMatricesBig) {
  lumin::Matrix A(1000, 1000), B(1000, 1000);
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = 1.0;
    B.data()[i] = 2.0;
  }
  lumin::Matrix C = A * B;

  EXPECT_EQ(C.rows(), 1000);
  EXPECT_EQ(C.cols(), 1000);

  EXPECT_EQ(C.data()[0], 2000);
  EXPECT_EQ(C.data()[1], 2000);
}

TEST_F(CPUMatrixTest, ScaleMatrix) {
  lumin::Matrix A(2,2);

  A.data()[0] = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  lumin::Matrix R = A * 2.0;

  EXPECT_EQ(R(0, 0), 2.0);
  EXPECT_EQ(R(1, 1), 8.0);
}

TEST_F(CPUMatrixTest, DotProductMatrices) {
  lumin::Matrix A(2,2), B(2,2);

  A(0, 0) = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  B.data()[0] = 5; B.data()[1] = 6;
  B.data()[2] = 7; B.data()[3] = 8;

  double R = A % B;

  EXPECT_EQ(R, 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8); 
}

TEST_F(CPUMatrixTest, SubtractMatrices) {
  lumin::Matrix A(2,2), B(2,2);

  A.data()[0] = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  B.data()[0] = 5; B.data()[1] = 5;
  B.data()[2] = 5; B.data()[3] = 5;

  lumin::Matrix C = B - A;

  EXPECT_EQ(C(0, 0), 4);
  EXPECT_EQ(C(1, 0), 2);
}

