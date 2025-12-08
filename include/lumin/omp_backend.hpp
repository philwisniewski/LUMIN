#pragma once
#include "backend.hpp"

#ifdef LUMIN_ENABLE_OPENMP
#include <omp.h>

namespace lumin {
  
  class OMPBackend : public Backend {
  public:
    Matrix add(const Matrix& A, const Matrix& B) override;
    Matrix multiply(const Matrix& A, const Matrix& B) override;
    Matrix subtract(const Matrix& A, const Matrix& B) override;
    Matrix scalar(double s, const Matrix& A) override;
    Matrix transpose(const Matrix& A) override;
    double dot(const Matrix& A, const Matrix& B) override;
    const char* name() const override { return "OPENMP"; }
  };

}

#endif // LUMIN_ENABLE_OPENMP