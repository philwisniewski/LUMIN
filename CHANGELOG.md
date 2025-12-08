# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-12-08

### Changed
- Added github repo

## [0.1.2] - 2025-12-08

### Added
- Documentation in README.md

## [0.1.1] - 2025-12-08

### Fixed
- Tests are no longer built during pip installation, fixing installation issues on macOS and other platforms
- Tests can still be built locally for development with `cmake -DENABLE_TESTS=ON`

## [0.1.0] - 2025-12-08

### Added
- Initial release of lumin-matrix
- High-performance matrix operations library with multiple backends:
  - CPU backend (always available)
  - OpenMP backend (if OpenMP is available)
  - CUDA backend (if CUDA toolkit is available)
  - MPI backend (if MPI is available)
- Python bindings via pybind11
- NumPy integration
- Matrix operations: addition, subtraction, multiplication, transpose, dot product
- Scalar multiplication
- Random matrix generation
- Backend selection API

### Installation
```bash
pip install lumin-matrix
```

### Usage
```python
import lumin

# Create matrices
A = lumin.Matrix(3, 3)
B = lumin.Matrix(3, 3)

# Matrix operations
C = A + B
D = A * B
E = A.transpose()

# Set backend
lumin.set_backend("cpu")  # or "openmp", "cuda", "mpi"
```

