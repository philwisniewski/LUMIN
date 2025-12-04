#pragma once
#include <memory>
#include "backend.hpp"

namespace lumin {

  class Matrix {
  public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, std::shared_ptr<Backend> backend);
    Matrix();

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    double* data() { return m_values.get(); }
    const double* data() const { return m_values.get(); }

    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(const Matrix& other) const;
    Matrix scalar(double s) const;
    Matrix transpose() const;
    double dot(const Matrix& other) const;

    double& operator()(size_t r, size_t c) { return m_values[r * m_cols + c]; }
    const double& operator()(size_t r, size_t c) const { return m_values[r * m_cols + c]; }

    Matrix operator+(const Matrix& other) const { return add(other); }
    Matrix operator-(const Matrix& other) const { return subtract(other); }
    Matrix operator*(const Matrix& other) const { return multiply(other); }
    Matrix operator*(double s) const { return scalar(s); }
    double operator%(const Matrix& other) const { return dot(other); }

    static Matrix random_int(size_t rows, size_t cols, int max_value);
    std::string to_string(int precision) const;

  private:
    size_t m_rows, m_cols;
    std::shared_ptr<Backend> backend;
    std::shared_ptr<double[]> m_values;
  };

}
