#pragma once
#include <memory>
#include "backend.hpp"

namespace lumin {

  class Matrix {
  public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, std::shared_ptr<Backend> backend);

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    double* data() { return m_values.get(); }
    const double* data() const { return m_values.get(); }

    Matrix add(const Matrix& other) const;
    Matrix multiply(const Matrix& other) const;
    Matrix scalar(double s) const;
    Matrix transpose() const;

  private:
    size_t m_rows, m_cols;
    std::shared_ptr<Backend> backend;
    std::shared_ptr<double[]> m_values;
  };

}
