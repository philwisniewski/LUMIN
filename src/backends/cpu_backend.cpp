#include "lumin.hpp"

namespace lumin {

static void check_same_size(const Matrix& A, const Matrix& B, const char* op) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    throw std::runtime_error("dimension mismatch in operation");
  }
}

static void check_multiply_dims(const Matrix& A, const Matrix& B) {
  if (A.cols() != B.rows()) {
    throw std::runtime_error("multiply dimension mismatch");
  }
}

Matrix CPUBackend::add(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "add");
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] + B.data()[i];
  }
  return R;
}

Matrix CPUBackend::subtract(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "subtract");
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] - B.data()[i];
  }
  return R;
}

Matrix CPUBackend::scalar(double s, const Matrix& A) {
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] * s;
  }
  return R;
}

Matrix CPUBackend::multiply(const Matrix& A, const Matrix& B) {
  check_multiply_dims(A, B);
  Matrix R(A.rows(), B.cols());
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t k = 0; k < A.cols(); k++) {
      double a = A(i, k);
      for (size_t j = 0; j < B.cols(); j++) {
        R(i, j) += a * B(k, j);
      }
    }
  }
  return R;
}

double CPUBackend::dot(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "dot");
  double res = 0.0;
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    res += A.data()[i] * B.data()[i];
  }
  return res;
}

Matrix CPUBackend::transpose(const Matrix& A) {
  Matrix R(A.cols(), A.rows());
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      R(j, i) = A(i, j);
    }
  }
  return R;
}

} // namespace lumin

