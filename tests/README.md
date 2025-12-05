# LUMIN Test Suite

This directory contains the test suite for LUMIN, organized by backend type.

## Test Structure

Tests are organized into separate files and executables for each backend:

- **`test_cpu.cpp`** - CPU-only tests (always built)
- **`test_mpi.cpp`** - MPI backend tests (built if MPI is enabled)
- **`test_cuda.cpp`** - CUDA backend tests (built if CUDA is enabled)
- **`test_openmp.cpp`** - OpenMP backend tests (ready for when OpenMP is implemented)
- **`test_utils.hpp`** - Common test utilities and helper functions

## Running Tests

### All Tests
```bash
cd build
ctest
```

### Specific Backend Tests
```bash
# CPU tests only
ctest -R test_cpu

# MPI tests only
ctest -R test_mpi

# CUDA tests only
ctest -R test_cuda
```

### Manual Execution
```bash
# CPU tests
./test_cpu

# MPI tests (requires mpiexec)
mpiexec -n 2 ./test_mpi

# CUDA tests
./test_cuda
```

## Test Organization

Each test file uses Google Test fixtures:
- `CPUMatrixTest` - CPU backend tests
- `MPIMatrixTest` - MPI backend tests
- `CUDAMatrixTest` - CUDA backend tests
- `OpenMPMatrixTest` - OpenMP backend tests (when implemented)

## Adding New Tests

1. **CPU Tests**: Add to `test_cpu.cpp` in the `CPUMatrixTest` fixture
2. **MPI Tests**: Add to `test_mpi.cpp` in the `MPIMatrixTest` fixture
3. **CUDA Tests**: Add to `test_cuda.cpp` in the `CUDAMatrixTest` fixture
4. **OpenMP Tests**: Add to `test_openmp.cpp` in the `OpenMPMatrixTest` fixture

## Test Utilities

The `test_utils.hpp` header provides helper functions:
- `create_sequential_matrix()` - Create matrix with sequential values
- `create_constant_matrix()` - Create matrix filled with constant value
- `matrices_equal()` - Compare matrices with tolerance
- `EXPECT_MATRIX_EQ()` - Google Test assertion macro for matrix equality
- `matrix_from_vector()` - Create matrix from 2D vector

## Conditional Compilation

Tests are conditionally compiled based on backend availability:
- CPU tests: Always built
- MPI tests: Built if `ENABLE_MPI=ON` and MPI is found
- CUDA tests: Built if `ENABLE_CUDA=ON` and CUDA toolkit is found
- OpenMP tests: Ready for when `ENABLE_OPENMP` option is added

If a backend is not available, the corresponding test suite will contain a skipped test indicating the backend is not enabled.

