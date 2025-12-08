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

__global__ void add_kernel(const double* A, const double* B, double* C, size_t M, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; i < M * N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

__global__ void subtract_kernel(const double* A, const double* B, double* C, size_t M, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; i < M * N; i += stride) {
    C[i] = A[i] - B[i];
  }
}

__global__ void scalar_kernel(const double* A, const double s, double* C, size_t M, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; i < M * N; i += stride) {
    C[i] = A[i] * s;
  }
}

__global__ void dot_product_kernel(const double* A, const double* B, double* C, size_t M, size_t N) {
  __shared__ float cache[256];

  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  double temp = 0.0f;
  size_t n = M * N;
  while (idx < n) {
    temp += A[idx] * B[idx];
    idx += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  size_t i = blockDim.x / 2;

  while (i != 0) {
    if (threadIdx.x < i) {
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }
  
  if (threadIdx.x == 0) {
    atomicAdd(C, cache[0]);
  }
}

__global__ void transpose_kernel(const double* A, double* C, size_t M, size_t N) {
  // A is M×N, C is N×M
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t stride_x = blockDim.x * gridDim.x;
  size_t stride_y = blockDim.y * gridDim.y;

  for (size_t i = row; i < N; i += stride_y) {
    for (size_t j = col; j < M; j += stride_x) {
      C[i * M + j] = A[j * N + i];
    }
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
  size_t M = A.rows();
  size_t N = A.cols();

  Matrix C(M, N);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  double *dA = deviceAllocCopy(A.data(), M * N * sizeof(double));
  double *dB = deviceAllocCopy(B.data(), M * N * sizeof(double));
  double *dC = nullptr;
  cudaMalloc(&dC, M * N * sizeof(double));

  add_kernel<<<grid, block>>>(dA, dB, dC, M, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return (M == 0 || N == 0) ? Matrix(0, 0) : C;
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
  size_t M = A.rows();
  size_t N = A.cols();

  Matrix C(M, N);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  double *dA = deviceAllocCopy(A.data(), M * N * sizeof(double));
  double *dB = deviceAllocCopy(B.data(), M * N * sizeof(double));
  double *dC = nullptr;
  cudaMalloc(&dC, M * N * sizeof(double));

  subtract_kernel<<<grid, block>>>(dA, dB, dC, M, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return (M == 0 || N == 0) ? Matrix(0, 0) : C;
}

Matrix CUDABackend::scalar(double s, const Matrix& A) {
  size_t M = A.rows();
  size_t N = A.cols();

  Matrix C(M, N);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  
  double *dA = deviceAllocCopy(A.data(), M * N * sizeof(double));
  double *dC = nullptr;
  cudaMalloc(&dC, M * N * sizeof(double));

  scalar_kernel<<<grid, block>>>(dA, s, dC, M, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dC);

  return (M == 0 || N == 0) ? Matrix(0, 0) : C;
}

Matrix CUDABackend::transpose(const Matrix& A) {
  size_t M = A.rows();
  size_t N = A.cols();

  Matrix C(N, M);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  
  double *dA = deviceAllocCopy(A.data(), M * N * sizeof(double));
  double *dC = nullptr;
  cudaMalloc(&dC, N * M * sizeof(double));

  transpose_kernel<<<grid, block>>>(dA, dC, M, N);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), dC, N * M * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dC);

  return (M == 0 || N == 0) ? Matrix(0, 0) : C;
}

double CUDABackend::dot(const Matrix& A, const Matrix& B) {
  size_t M = A.rows();
  size_t N = A.cols();

  double *dA = deviceAllocCopy(A.data(), M * N * sizeof(double));
  double *dB = deviceAllocCopy(B.data(), M * N * sizeof(double));
  double *dC = nullptr;
  cudaMalloc(&dC, sizeof(double));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  dot_product_kernel<<<grid, block>>>(dA, dB, dC, M, N);
  cudaDeviceSynchronize();

  double result;
  cudaMemcpy(&result, dC, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return (M == 0 || N == 0) ? 0.0 : result;
}

/* End class methods */

}
