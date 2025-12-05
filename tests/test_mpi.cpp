#include <gtest/gtest.h>
#include "lumin.hpp"
#ifdef LUMIN_ENABLE_MPI
#include <mpi.h>
#endif

#ifdef LUMIN_ENABLE_MPI

class MPIMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Ensure MPI is initialized (should be done in main)
    auto backend = lumin::create_mpi_backend(MPI_COMM_WORLD);
    lumin::set_default_backend(backend);
  }
};

TEST_F(MPIMatrixTest, AddMatrices) {
  auto b = lumin::create_mpi_backend(MPI_COMM_WORLD);
  lumin::set_default_backend(b);

  lumin::Matrix A(2,2), B(2,2);
  A.data()[0] = 1; A.data()[1] = 2; A.data()[2] = 3; A.data()[3] = 4;
  B.data()[0] = 5; B.data()[1] = 6; B.data()[2] = 7; B.data()[3] = 8;

  lumin::Matrix C = A + B; 

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    EXPECT_EQ(C(0,0), 6);
    EXPECT_EQ(C(1,1), 12);
  }
}

// Add more MPI-specific tests here

#else

// If MPI is not enabled, provide a dummy test to avoid empty test suite
TEST(MPIMatrixTest, DISABLED_MPINotEnabled) {
  GTEST_SKIP() << "MPI backend not enabled in this build";
}

#endif

int main(int argc, char **argv) {
#ifdef LUMIN_ENABLE_MPI
  MPI_Init(&argc, &argv);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  ::testing::InitGoogleTest(&argc, argv);
  
  // Suppress output from non-root ranks
  if (rank != 0) {
    ::testing::TestEventListeners& listeners = 
      ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }
  
  int result = RUN_ALL_TESTS();
  
  MPI_Finalize();
  return result;
#else
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif
}

