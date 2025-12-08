#include <gtest/gtest.h>
#include "lumin.hpp"
#ifdef LUMIN_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef LUMIN_ENABLE_OPENMP

class OMPMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Ensure OpenMP backend is used
    auto backend = lumin::create_omp_backend();
    lumin::set_default_backend(backend);
  }
};

TEST_F(OMPMatrixTest, ParallelAddMatrices) {
  // Example test - adjust based on your OpenMP backend implementation
  lumin::Matrix A(100, 100), B(100, 100);
  
  // Fill matrices
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = 1.0;
    B.data()[i] = 2.0;
  }
  
  lumin::Matrix C = A + B;
  
  // Verify result
  for (size_t i = 0; i < C.rows() * C.cols(); ++i) {
    EXPECT_EQ(C.data()[i], 3.0);
  }
}

TEST_F(OMPMatrixTest, ParallelMultiplyMatrices) {
  lumin::Matrix A(100, 100), B(100, 100);
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = 1.0;
    B.data()[i] = 2.0;
  }
  lumin::Matrix C = A * B;
  // For 100x100 matrices where A is all 1.0 and B is all 2.0:
  // Each element C[i][j] = sum over k of A[i][k] * B[k][j] = 100 * 1.0 * 2.0 = 200.0
  for (size_t i = 0; i < C.rows() * C.cols(); ++i) {
    EXPECT_EQ(C.data()[i], 200.0);
  }
}

TEST_F(OMPMatrixTest, ParallelSubtractMatrices) {
  lumin::Matrix A(100, 100), B(100, 100);
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = 1.0;
    B.data()[i] = 2.0;
  }
  lumin::Matrix C = A - B;
  for (size_t i = 0; i < C.rows() * C.cols(); ++i) {
    EXPECT_EQ(C.data()[i], -1.0);
  }
}

TEST_F(OMPMatrixTest, ParallelScalarMatrices) {

  lumin::Matrix A(100, 100);
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = 1.0;
  }
  lumin::Matrix C = A * 2.0;
  for (size_t i = 0; i < C.rows() * C.cols(); ++i) {
    EXPECT_EQ(C.data()[i], 2.0);
  }
}

TEST_F(OMPMatrixTest, ParallelTransposeMatrices) {
  lumin::Matrix A(100, 100);
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = i;
  }
  lumin::Matrix C = A.transpose();
  // For a 100x100 matrix, A.data()[i] = i means A(row, col) = row*100 + col
  // After transpose: C(row, col) = A(col, row) = col*100 + row
  // So C.data()[i] where i = row*100 + col should be col*100 + row
  for (size_t row = 0; row < C.rows(); ++row) {
    for (size_t col = 0; col < C.cols(); ++col) {
      size_t idx = row * C.cols() + col;
      double expected = col * A.cols() + row;  // A(col, row) = col*100 + row
      EXPECT_EQ(C.data()[idx], expected) << "At position (" << row << ", " << col << ")";
    }
  }
}

TEST_F(OMPMatrixTest, ParallelDotProductMatrices) {
  lumin::Matrix A(100, 100), B(100, 100);
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    A.data()[i] = 1.0;
    B.data()[i] = 2.0;
  }
  double C = A.dot(B);
  // Dot product = sum of all A[i] * B[i] = 10000 * 1.0 * 2.0 = 20000.0
  EXPECT_EQ(C, 20000.0);
}


#else

// If OpenMP is not enabled, provide a dummy test to avoid empty test suite
TEST(OMPMatrixTest, DISABLED_OpenMPNotEnabled) {
  GTEST_SKIP() << "OpenMP backend not enabled in this build";
}

#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

