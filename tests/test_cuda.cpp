#include <gtest/gtest.h>
#include "lumin.hpp"
#ifdef LUMIN_ENABLE_CUDA
// CUDA headers if needed
#endif

#ifdef LUMIN_ENABLE_CUDA

class CUDAMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Ensure CUDA backend is used
    auto backend = lumin::create_cuda_backend();
    lumin::set_default_backend(backend);
  }
};

TEST_F(CUDAMatrixTest, MultiplyMatrices) {
  auto b = lumin::create_cuda_backend();
  lumin::set_default_backend(b);

  lumin::Matrix A(2, 3), B(3, 1);
  A.data()[0] = 1; A.data()[1] = 2; A.data()[2] = 3;
  A.data()[3] = 4; A.data()[4] = 5; A.data()[5] = 6;

  B.data()[0] = 7;
  B.data()[1] = 8;
  B.data()[2] = 9;

  lumin::Matrix C = A * B;

  EXPECT_EQ(C.rows(), 2);
  EXPECT_EQ(C.cols(), 1);

  EXPECT_EQ(C.data()[0], 50);
  EXPECT_EQ(C.data()[1], 122);
}

TEST_F(CUDAMatrixTest, MultiplyMatricesBig) {
  auto b = lumin::create_cuda_backend();
  lumin::set_default_backend(b);
  
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

#else

// If CUDA is not enabled, provide a dummy test to avoid empty test suite
TEST(CUDAMatrixTest, DISABLED_CUDANotEnabled) {
  GTEST_SKIP() << "CUDA backend not enabled in this build";
}

#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

