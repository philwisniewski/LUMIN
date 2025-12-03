#pragma once
#include "backend.hpp"

namespace lumin {

class CUDABackend : public Backend {
public:
  CUDABackend(int device_id = 0);

  Matrix add(const Matrix& A, const Matrix& B) override;
  Matrix multiply(const Matrix& A, const Matrix& B) override;
  Matrix subtract(const Matrix& A, const Matrix& B) override;
  Matrix scalar(double s, const Matrix& A) override;
  Matrix transpose(const Matrix& A) override;

  const char* name() const override { return "CUDA"; }

private:
  int device;
};

}
