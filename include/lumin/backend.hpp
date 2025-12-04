#pragma once
#include <memory>

namespace lumin {

  class Matrix;

  class Backend {
  public:
    virtual ~Backend() = default;

    virtual Matrix add(const Matrix& A, const Matrix& B) = 0;
    virtual Matrix multiply(const Matrix& A, const Matrix& B) = 0;
    virtual Matrix subtract(const Matrix& A, const Matrix& B) = 0;
    virtual Matrix scalar(double s, const Matrix& A) = 0;
    virtual Matrix transpose(const Matrix& A) = 0;
    virtual double dot(const Matrix& A, const Matrix& B) = 0;

    virtual const char* name() const = 0;
  };

}
