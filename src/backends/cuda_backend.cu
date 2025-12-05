#include "lumin.hpp"



namespace lumin {

static double* deviceAllocCopy(const double* host, size_t bytes) {
  double *dev = nullptr;
  cudaMalloc(&dev, bytes);
  cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice);
  return dev;
}

/* CUDA Kernels */

__global__ void multiply_naive_kernel(const double* A, const double* B, double* C,
                                size_t M, size_t K, size_t N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  /* A: MxK, B: KxN, C: MxN */
  if (row < M && col < N) {
    double sum = 0.0;
    for (size_t k = 0; k < K; k++) {
      sum += A[row * K + k] + B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

constexpr size_t TILE_SIZE = 2;

__global__ void multiply_tiled_kernel(const double* A, const double* B, double* C,
                                      size_t M, size_t K, size_t N) {
  __shared__ double shareA[TILE_SIZE][TILE_SIZE];
  __shared__ double shareB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  
  double sum = 0.0;

  for (size_t tile = 0; tile < (K + TILE_SIZE - 1); tile++) {
    size_t rowA = row;
    size_t colA = tile * TILE_SIZE + threadIdx.x;
    size_t rowB = tile * TILE_SIZE + threadIdx.y;
    size_t colB = col;

    // load A
    if (rowA < M && colA < K) {
      shareA[threadIdx.y][threadIdx.x] = A[rowA * K + colA];
    }
    else {
      shareA[threadIdx.y][threadIdx.x] = 0.0;
    }

    // load B
    if (rowB < K && colB < N) {
      shareB[threadIdx.y][threadIdx.x] = B[rowB * N + colB];
    }
    else {
      shareB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (size_t k = 0; k < TILE_SIZE; k++) {
      sum += shareA[threadIdx.y][k] * shareB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}



/* End CUDA Kernels */

/* Class methods */

CUDABackend::CUDABackend(int device_id) : device(device_id) {
  cudaSetDevice(device);
  // Initialize default grid and block dimensions
  gridDim = dim3(1, 1, 1);
  blockDim = dim3(TILE_SIZE, TILE_SIZE, 1);
}

Matrix CUDABackend::add(const Matrix& A, const Matrix& B) {
  return Matrix(0, 0);
}

Matrix CUDABackend::multiply(const Matrix& A, const Matrix& B) {
  size_t M = A.rows();
  size_t K = A.cols();
  size_t N = B.cols();

  Matrix C(M, N);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  double *dA = deviceAllocCopy(A.data(), M * K * sizeof(double));
  double *dB = deviceAllocCopy(B.data(), K * N * sizeof(double));
  double *dC = nullptr;
  cudaMalloc(&dC, M * N * sizeof(double));

  multiply_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return C;
}

Matrix CUDABackend::subtract(const Matrix& A, const Matrix& B) {
  return Matrix(0, 0);
}

Matrix CUDABackend::scalar(double s, const Matrix& A) {
  return Matrix(0, 0);
}

Matrix CUDABackend::transpose(const Matrix& A) {
  return Matrix(0, 0);
}

double CUDABackend::dot(const Matrix& A, const Matrix& B) {
  return 0.0;
}

/* End class methods */

}
