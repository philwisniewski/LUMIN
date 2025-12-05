#ifndef LUMIN_TEST_UTILS_HPP
#define LUMIN_TEST_UTILS_HPP

#include "lumin.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

namespace lumin_test {

// Helper function to create a test matrix with sequential values
inline lumin::Matrix create_sequential_matrix(size_t rows, size_t cols, double start = 0.0) {
  lumin::Matrix m(rows, cols);
  double val = start;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      m.data()[i * cols + j] = val++;
    }
  }
  return m;
}

// Helper function to create a matrix filled with a constant value
inline lumin::Matrix create_constant_matrix(size_t rows, size_t cols, double value) {
  lumin::Matrix m(rows, cols);
  for (size_t i = 0; i < rows * cols; ++i) {
    m.data()[i] = value;
  }
  return m;
}

// Helper function to compare two matrices with tolerance
inline bool matrices_equal(const lumin::Matrix& A, const lumin::Matrix& B, double tolerance = 1e-9) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    return false;
  }
  
  for (size_t i = 0; i < A.rows() * A.cols(); ++i) {
    if (std::abs(A.data()[i] - B.data()[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

// Google Test assertion macro for matrix equality
#define EXPECT_MATRIX_EQ(A, B, tolerance) \
  EXPECT_TRUE(lumin_test::matrices_equal(A, B, tolerance)) \
      << "Matrix A:\n" << #A << "\nMatrix B:\n" << #B

// Helper to initialize a matrix from a 2D vector
inline lumin::Matrix matrix_from_vector(const std::vector<std::vector<double>>& data) {
  if (data.empty()) {
    return lumin::Matrix(0, 0);
  }
  
  size_t rows = data.size();
  size_t cols = data[0].size();
  lumin::Matrix m(rows, cols);
  
  for (size_t i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      throw std::invalid_argument("All rows must have the same number of columns");
    }
    for (size_t j = 0; j < cols; ++j) {
      m.data()[i * cols + j] = data[i][j];
    }
  }
  return m;
}

} // namespace lumin_test

#endif // LUMIN_TEST_UTILS_HPP

