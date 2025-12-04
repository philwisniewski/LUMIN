#include "lumin.hpp"
#include "lumin.hpp"

#include <memory>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace lumin {

static std::shared_ptr<double[]> allocate_buffer(size_t n) {
  // return std::shared_ptr<double>(new double[n](), [](double* p){ delete[] p; });
  return std::shared_ptr<double[]>(new double[n], std::default_delete<double[]>());
}

Matrix::Matrix(size_t rows, size_t cols)
  : m_rows(rows), m_cols(cols),
    backend(get_default_backend()), // backend(nullptr),
    m_values( allocate_buffer(rows * cols) )
{ }

Matrix::Matrix(size_t rows, size_t cols, std::shared_ptr<Backend> backend_ptr)
  : m_rows(rows), m_cols(cols),
    backend(std::move(backend_ptr)),
    m_values( allocate_buffer(rows * cols) )
{ }

Matrix::Matrix()
  : m_rows(0), m_cols(0), backend(nullptr), m_values(nullptr)
{ }

//  double* Matrix::data() noexcept {
//   return m_values.get();
// }

// const double* Matrix::data() const noexcept {
//   return m_values.get();
// }

static void check_same_size(const Matrix& A, const Matrix& B, const char* op) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    std::ostringstream oss;
    oss << "Matrix " << op << " dimension mismatch: "
        << "(" << A.rows() << "x" << A.cols() << ") vs "
        << "(" << B.rows() << "x" << B.cols() << ")";
    throw std::runtime_error(oss.str());
  }
}

static void check_multiply_dims(const Matrix& A, const Matrix& B) {
  if (A.cols() != B.rows()) {
    std::ostringstream oss;
    oss << "Matrix multiply dimension mismatch: "
        << "(" << A.rows() << "x" << A.cols() << ") vs "
        << "(" << B.rows() << "x" << B.cols() << ")";
    throw std::runtime_error(oss.str());
  }
}

// CPU fallback
Matrix cpu_add(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "add");
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] + B.data()[i];
  }
  return R;
}

Matrix cpu_subtract(const Matrix& A, const Matrix& B) {
  check_same_size(A, B, "subtract");
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] - B.data()[i];
  }
  return R;
}

Matrix cpu_scalar(double s, const Matrix& A) {
  Matrix R(A.rows(), A.cols());
  size_t N = A.rows() * A.cols();
  for (size_t i = 0; i < N; i++) {
    R.data()[i] = A.data()[i] * s;
  }
  return R;
}

Matrix cpu_multiply(const Matrix& A, const Matrix& B) {
  check_multiply_dims(A, B);
  Matrix R(A.rows(), B.cols());
  for (size_t i = 0; i < static_cast<size_t>(A.rows()); i++) {
    for (size_t k = 0; k < static_cast<size_t>(A.cols()); k++) {
      double a = A.data()[i * A.cols() + k];
      size_t rowR = i * R.cols();
      size_t rowB = k * B.cols();
      for (size_t j = 0; j < static_cast<size_t>(B.cols()); j++) {
        R.data()[rowR + j] += a * B.data()[rowB + j];
      }
    }
  }
  return R;
}

double cpu_dot(const Matrix& A, const Matrix& B) {
  double res = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(A.rows()); i++) {
    for (size_t j = 0; j < static_cast<size_t>(A.cols()); j++) {
      res += A(i, j) * B(i, j);
    }
  }
  return res;
}

Matrix cpu_transpose(const Matrix& A) {
  Matrix R(A.cols(), A.rows());
  for (size_t i = 0; i < static_cast<size_t>(A.rows()); i++) {
    for (size_t j = 0; j < static_cast<size_t>(A.cols()); j++) {
      R.data()[j * R.cols() + i] = A.data()[i * A.cols() + j];
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

double Matrix::dot(const Matrix& other) const {
  if (backend) {
    return backend->dot(*this, other);
  }
  return cpu_dot(*this, other);
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
    R.data()[i] = static_cast<double>(dis(gen));
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
