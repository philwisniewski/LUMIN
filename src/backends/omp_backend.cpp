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

Matrix OMPBackend::add(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "add");
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();

  #pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] + B.data()[i];
  }
  return R;
}

Matrix OMPBackend::subtract(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "subtract");
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();

  #pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] - B.data()[i];
  }
  return R;
}

Matrix OMPBackend::scalar(double s, const Matrix& A) {
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  
  #pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] * s;
  }
  return R;
}

Matrix OMPBackend::multiply(const Matrix& A, const Matrix& B) {
  check_multiply_dims(A, B);
  Matrix R(A.rows(), B.cols());

  #pragma omp parallel for
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

double OMPBackend::dot(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "dot");
  double res = 0.0;
  size_t N = A.rows() * A.cols();

  #pragma omp parallel for reduction(+:res)
  for (size_t i = 0; i < N; i++) {
    res += A.data()[i] * B.data()[i];
  }
  return res;
}

Matrix OMPBackend::transpose(const Matrix& A) {
  Matrix R(A.cols(), A.rows());

  #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < A.cols(); j++) {
      R(j, i) = A(i, j);
    }
  }
  return R;
}

} // namespace lumin

