#pragma once
#include "backend.hpp"

#ifdef LUMIN_ENABLE_CUDA
#include <cuda_runtime.h>

namespace lumin {

class CUDABackend : public Backend {
public:
  CUDABackend(int device_id = 0);

  Matrix add(const Matrix& A, const Matrix& B) override;
  Matrix multiply(const Matrix& A, const Matrix& B) override;
  Matrix subtract(const Matrix& A, const Matrix& B) override;
  Matrix scalar(double s, const Matrix& A) override;
  Matrix transpose(const Matrix& A) override;
  double dot(const Matrix& A, const Matrix& B) override;

  const char* name() const override { return "CUDA"; }

private:
  int device;
  dim3 gridDim;
  dim3 blockDim;
};

} // namespace lumin

#endif // LUMIN_ENABLE_CUDA
