#include <gtest/gtest.h>
#include "lumin.hpp"
#include <mpi.h>
#include <string>

TEST(MatrixTest, CreateAndFill) {
  lumin::Matrix A(3,3);
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = 0; j < A.cols(); ++j) {
      A.data()[i * A.cols() + j] = i + j;
    }
  }

  EXPECT_EQ(A.data()[0], 0);
  EXPECT_EQ(A.data()[2*3 + 1], 3); // row 2, col 1
}

TEST(MatrixTest, AddMatrices) {
  lumin::Matrix A(2,2), B(2,2);

  A.data()[0] = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  B.data()[0] = 5; B.data()[1] = 6;
  B.data()[2] = 7; B.data()[3] = 8;

  lumin::Matrix C = A.add(B);

  EXPECT_EQ(C.data()[0], 6);
  EXPECT_EQ(C.data()[3], 12);
}

TEST(MatrixTest, MultiplyMatrices) {
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

TEST(MatrixTest, ScaleMatrix) {
  lumin::Matrix A(2,2);

  A.data()[0] = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  lumin::Matrix R = A * 2.0;

  EXPECT_EQ(R(0, 0), 2.0);
  EXPECT_EQ(R(1, 1), 8.0);
}

TEST(MatrixTest, DotProductMatrices) {
  lumin::Matrix A(2,2), B(2,2);

  A(0, 0) = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  B.data()[0] = 5; B.data()[1] = 6;
  B.data()[2] = 7; B.data()[3] = 8;

  double R = A % B;

  EXPECT_EQ(R, 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8); 
}

TEST(MatrixTest, SubtractMatrices) {
  lumin::Matrix A(2,2), B(2,2);

  A.data()[0] = 1; A.data()[1] = 2;
  A.data()[2] = 3; A.data()[3] = 4;

  B.data()[0] = 5; B.data()[1] = 5;
  B.data()[2] = 5; B.data()[3] = 5;

  lumin::Matrix C = B - A;

  EXPECT_EQ(C(0, 0), 4);
  EXPECT_EQ(C(1, 0), 2);
}

/* mpi backend tests? */

TEST(MPI_Test, AddMatrices) {
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

int main(int argc, char **argv) {
  std::string mode = (argc > 1 ? argv[1] : "cpu");

  if (mode == "mpi") {
    MPI_Init(&argc, &argv);
  }

  ::testing::InitGoogleTest(&argc, argv);

  int rank = 0;

  if (mode == "mpi") {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
    if (rank != 0) {
      ::testing::TestEventListeners& listeners = 
        ::testing::UnitTest::GetInstance()->listeners();
      delete listeners.Release(listeners.default_result_printer());
    }
  }

  if (mode == "mpi") {
    ::testing::GTEST_FLAG(filter) = "MPI_Test.*";
    if (rank != 0) {
      ::testing::GTEST_FLAG(output) = "none";
    }
  } else {
    ::testing::GTEST_FLAG(filter) = "-MPI_Test.*"; 
  }

  int result = RUN_ALL_TESTS();

  if (mode == "mpi") {
    MPI_Finalize();
  }

  return result;
}

