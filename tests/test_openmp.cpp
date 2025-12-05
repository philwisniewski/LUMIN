#include <gtest/gtest.h>
#include "lumin.hpp"
#ifdef LUMIN_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef LUMIN_ENABLE_OPENMP

class OpenMPMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Ensure OpenMP backend is used
    // Note: You'll need to implement create_openmp_backend() in your factory
    // auto backend = lumin::create_openmp_backend();
    // lumin::set_default_backend(backend);
    
    // For now, use CPU backend but with OpenMP enabled
    auto backend = lumin::create_cpu_backend();
    lumin::set_default_backend(backend);
  }
};

TEST_F(OpenMPMatrixTest, ParallelAddMatrices) {
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

// Add more OpenMP-specific tests here

#else

// If OpenMP is not enabled, provide a dummy test to avoid empty test suite
TEST(OpenMPMatrixTest, DISABLED_OpenMPNotEnabled) {
  GTEST_SKIP() << "OpenMP backend not enabled in this build";
}

#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

