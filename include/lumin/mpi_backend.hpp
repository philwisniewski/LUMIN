#pragma once
#include "backend.hpp"

namespace lumin {
  
  class MPIBackend : public Backend {
  public:
    MPIBackend(int rank, int size);

    Matrix add(const Matrix& A, const Matrix& B) override;
    Matrix multiply(const Matrix& A, const Matrix& B) override;
    Matrix subtract(const Matrix& A, const Matrix& B) override;
    Matrix scalar(double s, const Matrix& A) override;
    Matrix transpose(const Matrix& A) override;

    const char* name() const override { return "MPI"; }

  private:
    int rank, size;
  };

}
