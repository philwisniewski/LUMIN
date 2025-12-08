# LUMIN Python Bindings

Python bindings for the LUMIN high-performance matrix operations library.

## Building

### Using CMake (Recommended)

```bash
cd build
cmake .. -DENABLE_PYTHON=ON
make lumin_python
```

The module will be built as `lumin.so` (or `lumin.pyd` on Windows) in the build directory.

### Using setup.py

```bash
pip install pybind11 numpy
python setup.py build_ext --inplace
```

Or install directly:

```bash
pip install .
```

## Requirements

- Python 3.6+
- NumPy
- pybind11 (>=2.6.0)
- C++17 compatible compiler

## Usage

```python
import numpy as np
import lumin

# Create matrices
A = lumin.Matrix(3, 3)
B = lumin.Matrix(3, 3)

# Fill with values
for i in range(3):
    for j in range(3):
        A[i, j] = i + j
        B[i, j] = 2 * (i + j)

# Matrix operations
C = A + B
D = A * B
E = A.transpose()
dot = A.dot(B)

# Create from NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
M = lumin.Matrix(arr)

# Convert back to NumPy
result = M.to_numpy()

# Set backend
lumin.set_backend("cpu")      # CPU backend
lumin.set_backend("openmp")   # OpenMP backend (if available)
lumin.set_backend("cuda")     # CUDA backend (if available)
lumin.set_backend("mpi")      # MPI backend (if available)

# Create random matrix
R = lumin.Matrix.random_int(10, 10, max_value=100)
```

## API Reference

### Matrix Class

#### Constructors
- `Matrix()` - Create empty matrix
- `Matrix(rows, cols)` - Create matrix with specified dimensions
- `Matrix(numpy_array)` - Create matrix from NumPy array

#### Properties
- `rows()` - Get number of rows
- `cols()` - Get number of columns
- `shape` - Get (rows, cols) tuple

#### Operations
- `add(other)` - Add another matrix
- `subtract(other)` - Subtract another matrix
- `multiply(other)` - Matrix multiplication
- `scalar(s)` - Multiply by scalar
- `transpose()` - Transpose matrix
- `dot(other)` - Dot product

#### Operators
- `A + B` - Matrix addition
- `A - B` - Matrix subtraction
- `A * B` - Matrix multiplication
- `A * s` or `s * A` - Scalar multiplication
- `A % B` - Dot product
- `A[i, j]` - Element access

#### Static Methods
- `Matrix.random_int(rows, cols, max_value=100)` - Create random integer matrix

### Backend Functions

- `create_cpu_backend()` - Create CPU backend
- `create_omp_backend()` - Create OpenMP backend (if available)
- `create_cuda_backend()` - Create CUDA backend (if available)
- `create_mpi_backend(comm=0)` - Create MPI backend (if available)
- `set_default_backend(backend)` - Set default backend
- `get_default_backend()` - Get current default backend
- `set_backend(name)` - Set backend by name ("cpu", "openmp", "cuda", "mpi")

## Examples

See the `examples/` directory for more detailed examples.

