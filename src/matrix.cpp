#include "lumin/matrix.hpp"
#include "lumin/backend.hpp"

#include <memory>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace lumin {

static std::shared_ptr<double> allocate_buffer(size_t n) {
  return std::shared_ptr<double>(new double[n](), [](double* p){ delete[] p; });
}

Matrix::Matrix(size_t rows, size_t cols)
  : m_rows(rows), m_cols(cols),
    backend(nullptr),
    m_values( allocate_buffer(rows * cols) )
{ }

Matrix::Matrix(size_t rows, size_t cols, std::shared_ptr<Backend> backend_ptr)
  : m_rows(rows), m_cols(cols),
    backend(std::move(backend_ptr)),
    m_values( allocate_buffer(rows * cols) )
{ }

double* Matrix::data() noexcept {
  return m_values.get();
}

const double* Matrix::data() const noexcept {
  return m_values.get();
}

static void check_same_size(const Matrix& A, const Matrix& B, const char* op) {
  if (A.m_rows != B.m_rows || A.m_cols != B.m_cols) {
    std::ostringstream oss;
    oss << "Matrix " << op << " dimension mismatch: "
        << "(" << A.m_rows << "x" << A.m_cols << ") vs "
        << "(" << B.m_rows << "x" << B.m_cols << ")";
    throw std::runtime_error(oss.str());
  }
}

static void check_multiply_dims(const Matrix& A, const Matrix& B) {
  if (A.m_cols != B.m_rows) {
    std::ostringstream oss;
    oss << "Matrix multiply dimension mismatch: "
        << "(" << A.m_rows << "x" << A.m_cols << ") vs "
        << "(" << B.m_rows << "x" << B.m_cols << ")";
    throw std::runtime_error(oss.str());
  }
}

// CPU fallback
Matrix cpu_add(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "add");
  Matrix R(A.m_rows, A.m_cols);
  size_t N = A.m_rows * A.m_cols;
  for (size_t i = 0; i < N; i++) {
    R.m_values.get()[i] = A.m_values.get()[i] + B.m_values.get()[i];
  }
  return R;
}

Matrix cpu_subtract(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "subtract");
  Matrix R(A.m_rows, A.m_cols);
  size_t N = A.m_rows * A.m_cols;
  for (size_t i = 0; i < N; i++) {
    R.m_values.get()[i] = A.m_values.get()[i] - B.m_values.get()[i];
  }
  return R;
}

Matrix cpu_scalar(double s, const Matrix& A) {
  Matrix R(A.m_rows, A.m_cols);
  size_t N = A.m_rows * A.m_cols;
  for (size_t i = 0; i < N; i++) {
    R.m_values.get()[i] = A.m_values.get()[i] * s;
  }
  return R;
}

Matrix cpu_multiply(const Matrix& A, const Matrix& B) {
  check_multiply_dims(A, B);
  Matrix R(A.m_rows, B.m_cols);
  for (size_t i = 0; i < static_cast<size_t>(A.m_rows); i++) {
    for (size_t k = 0; k < static_cast<size_t>(A.m_cols); k++) {
      double a = A.m_values.get()[i * A.m_cols + k];
      size_t rowR = i * R.m_cols;
      size_t rowB = k * B.m_cols;
      for (size_t j = 0; j < static_cast<size_t>(B.m_cols); j++) {
        R.m_values.get()[rowR + j] += a * B.m_values.get()[rowB + j];
      }
    }
  }
  return R;
}

Matrix cpu_transpose(const Matrix& A) {
  Matrix R(A.m_cols, A.m_rows);
  for (size_t i = 0; i < static_cast<size_t>(A.m_rows); i++) {
    for (size_t j = 0; j < static_cast<size_t>(A.m_cols); j++) {
      R.m_values.get()[j * R.m_cols + i] = A.m_values.get()[i * A.m_cols + j];
    }
  }
  return R;
}

// public API
Matrix Matrix::add(const Matrix& other) const {
  if (backend) {
    return backend->add(*this, other);
  }
  return cpu_add(*this, other);
}

Matrix Matrix::subtract(const Matrix& other) const {
  if (backend) {
    return backend->subtract(*this, other);
  }
  return cpu_subtract(*this, other);
}

Matrix Matrix::scalar(double s) const {
  if (backend) {
    return backend->scalar(s, *this);
  }
  return cpu_scalar(s, *this);
}

Matrix Matrix::multiply(const Matrix& other) const {
  if (backend) {
    return backend->multiply(*this, other);
  }
  return cpu_multiply(*this, other);
}

Matrix Matrix::transpose() const {
  if (backend) {
    return backend->transpose(*this);
  }
  return cpu_transpose(*this);
}

Matrix Matrix::random_int(size_t rows, size_t cols, int max_value) {
  Matrix R(rows, cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, max_value);
  size_t N = rows * cols;
  for (size_t i = 0; i < N; i++) {
    R.m_values.get()[i] = static_cast<double>(dis(gen));
  }
  return R;
}

std::string Matrix::to_string(int precision) const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      oss << m_values.get()[i * m_cols + j];
      if (j + 1 < m_cols) {
        oss << " ";
      }
    }
    if (i + 1 < m_rows) {
      oss << "\n";
    }
  }
  return oss.str();
}

}
