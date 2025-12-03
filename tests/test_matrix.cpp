#include <gtest/gtest.h>
#include "lumin/matrix.hpp"

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

