# LUMIN

**Library for Unified Matrix INfrastructure** - A high-performance matrix operations library with multiple backend support (CPU, OpenMP, CUDA, MPI).

[![PyPI version](https://img.shields.io/pypi/v/lumin-matrix.svg)](https://pypi.org/project/lumin-matrix/)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

## Features

- 🚀 **High Performance**: Optimized C++ implementation with multiple backend support
- 🔧 **Multiple Backends**: CPU, OpenMP, CUDA, and MPI backends
- 🐍 **Python Bindings**: Easy-to-use Python API via pybind11
- 🔄 **NumPy Integration**: Seamless conversion to/from NumPy arrays
- ⚡ **Flexible**: Choose the best backend for your hardware and workload

## Installation

```bash
pip install lumin-matrix
```

### Requirements

- Python 3.6+
- NumPy
- C++17 compatible compiler (for building from source)

Optional dependencies (for specific backends):
- OpenMP (for parallel CPU operations)
- CUDA Toolkit (for GPU acceleration)
- MPI (for distributed computing)

## Quick Start

```python
import lumin
import numpy as np

# Create matrices
A = lumin.Matrix(3, 3)
B = lumin.Matrix(3, 3)

# Fill with values
for i in range(3):
    for j in range(3):
        A[i, j] = i * 3 + j + 1
        B[i, j] = (i * 3 + j + 1) * 2

# Matrix operations
C = A + B          # Addition
D = A * B          # Matrix multiplication
E = A.transpose()  # Transpose
dot = A.dot(B)     # Dot product

# Scalar operations
F = A * 2.5        # Scalar multiplication
G = 3.0 * A       # Right-side scalar multiplication

# NumPy integration
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
M = lumin.Matrix(arr)      # Convert NumPy array to LUMIN Matrix
result = M.to_numpy()       # Convert back to NumPy

# Random matrix
R = lumin.Matrix.random_int(5, 5, max_value=100)

# Set backend
lumin.set_backend("cpu")     # CPU backend (always available)
lumin.set_backend("openmp")  # OpenMP backend (if available)
lumin.set_backend("cuda")    # CUDA backend (if available)
lumin.set_backend("mpi")     # MPI backend (if available)
```

## API Reference

### Matrix Class

#### Constructors

- `Matrix()` - Create empty matrix
- `Matrix(rows, cols)` - Create matrix with specified dimensions (filled with zeros)
- `Matrix(numpy_array)` - Create matrix from NumPy array

#### Properties

- `rows()` - Get number of rows
- `cols()` - Get number of columns
- `shape` - Get (rows, cols) tuple

#### Methods

- `add(other)` - Add another matrix
- `subtract(other)` - Subtract another matrix
- `multiply(other)` - Matrix multiplication
- `scalar(s)` - Multiply by scalar
- `transpose()` - Transpose the matrix
- `dot(other)` - Compute dot product with another matrix
- `to_numpy()` - Convert matrix to NumPy array

#### Operators

- `A + B` - Matrix addition
- `A - B` - Matrix subtraction
- `A * B` - Matrix multiplication
- `A * s` or `s * A` - Scalar multiplication
- `A % B` - Dot product
- `A[i, j]` - Element access (get/set)

#### Static Methods

- `Matrix.random_int(rows, cols, max_value=100)` - Create matrix with random integer values

### Backend Functions

- `create_cpu_backend()` - Create CPU backend
- `create_omp_backend()` - Create OpenMP backend (if available)
- `create_cuda_backend()` - Create CUDA backend (if available)
- `create_mpi_backend(comm=0)` - Create MPI backend (if available)
- `set_default_backend(backend)` - Set default backend
- `get_default_backend()` - Get current default backend
- `set_backend(name)` - Set backend by name ("cpu", "openmp", "cuda", "mpi")

## Backends

### CPU Backend
Always available. Single-threaded CPU operations.

```python
lumin.set_backend("cpu")
```

### OpenMP Backend
Parallel CPU operations using OpenMP. Automatically enabled if OpenMP is available.

```python
lumin.set_backend("openmp")
```

### CUDA Backend
GPU acceleration using NVIDIA CUDA. Requires CUDA toolkit and compatible GPU.

```python
lumin.set_backend("cuda")
```

### MPI Backend
Distributed computing using MPI. Requires MPI library (e.g., OpenMPI, MPICH).

```python
lumin.set_backend("mpi")
```

## Examples

See [`python/example.py`](python/example.py) for a complete example.

### Basic Operations

```python
import lumin

# Create and fill matrices
A = lumin.Matrix(2, 2)
A[0, 0] = 1.0
A[0, 1] = 2.0
A[1, 0] = 3.0
A[1, 1] = 4.0

B = lumin.Matrix(2, 2)
B[0, 0] = 5.0
B[0, 1] = 6.0
B[1, 0] = 7.0
B[1, 1] = 8.0

# Operations
C = A + B
D = A * B
E = A.transpose()
```

### NumPy Integration

```python
import numpy as np
import lumin

# Create from NumPy
arr = np.random.rand(100, 100)
matrix = lumin.Matrix(arr)

# Convert back
result = matrix.to_numpy()
```

### Backend Selection

```python
import lumin

# Try different backends
backends = ["cpu", "openmp", "cuda", "mpi"]

for backend_name in backends:
    try:
        lumin.set_backend(backend_name)
        print(f"✓ {backend_name} backend available")
    except Exception as e:
        print(f"✗ {backend_name} backend not available: {e}")
```

## Building from Source

### Prerequisites

- CMake 3.16+
- C++17 compatible compiler
- Python 3.6+
- NumPy
- pybind11

Optional:
- OpenMP
- CUDA Toolkit
- MPI (OpenMPI or MPICH)

### Build Steps

```bash
# Clone repository
git clone <repository-url>
cd LUMIN

# Install build dependencies
pip install scikit-build-core pybind11 numpy

# Build and install
pip install -e .

# Or build wheel
python -m build
```

### Building Tests

Tests are disabled by default. To build and run tests:

```bash
mkdir build
cd build
cmake .. -DENABLE_TESTS=ON
make
ctest
```

## Development

### Project Structure

```
LUMIN/
├── include/          # C++ headers
│   └── lumin/
├── src/             # C++ source files
│   └── backends/
├── python/          # Python bindings
│   ├── bindings.cpp
│   └── example.py
├── tests/           # Test suite
├── CMakeLists.txt    # CMake configuration
├── pyproject.toml   # Python package configuration
└── setup.py         # Setup script
```

### Running Tests

```bash
cd build
ctest                    # Run all tests
ctest -R test_cpu        # Run CPU tests only
ctest -R test_cuda       # Run CUDA tests only
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes in each version.

## Links

- [PyPI Package](https://pypi.org/project/lumin-matrix/)
- [GitHub Repository](https://github.com/philwisniewski/LUMIN)

## Author

Philip Wisniewski
