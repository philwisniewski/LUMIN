#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "lumin.hpp"
#include <vector>
#include <sstream>

namespace py = pybind11;
using namespace lumin;

// Helper function to create Matrix from numpy array
Matrix matrix_from_numpy(py::array_t<double> arr) {
    py::buffer_info buf_info = arr.request();
    
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    size_t rows = buf_info.shape[0];
    size_t cols = buf_info.shape[1];
    double* data = static_cast<double*>(buf_info.ptr);
    
    Matrix m(rows, cols);
    std::copy(data, data + rows * cols, m.data());
    
    return m;
}

// Helper function to convert Matrix to numpy array
py::array_t<double> matrix_to_numpy(const Matrix& m) {
    auto result = py::array_t<double>({m.rows(), m.cols()});
    py::buffer_info buf_info = result.request();
    double* ptr = static_cast<double*>(buf_info.ptr);
    
    std::copy(m.data(), m.data() + m.rows() * m.cols(), ptr);
    
    return result;
}

PYBIND11_MODULE(lumin, m) {
    m.doc() = "LUMIN: High-performance matrix operations library with multiple backends";

    // Matrix class
    py::class_<Matrix>(m, "Matrix")
        // Constructors
        .def(py::init<>())
        .def(py::init<size_t, size_t>(), 
             py::arg("rows"), py::arg("cols"),
             "Create a matrix with specified dimensions")
        .def(py::init(&matrix_from_numpy),
             py::arg("array"),
             "Create a matrix from a numpy array")
        
        // Properties
        .def("rows", &Matrix::rows, "Get number of rows")
        .def("cols", &Matrix::cols, "Get number of columns")
        .def("shape", [](const Matrix& m) {
            return std::make_pair(m.rows(), m.cols());
        }, "Get matrix shape as (rows, cols) tuple")
        
        // Element access
        .def("__getitem__", [](const Matrix& m, std::pair<size_t, size_t> idx) {
            return m(idx.first, idx.second);
        }, py::arg("index"), "Get element at (row, col)")
        .def("__setitem__", [](Matrix& m, std::pair<size_t, size_t> idx, double val) {
            m(idx.first, idx.second) = val;
        }, py::arg("index"), py::arg("value"), "Set element at (row, col)")
        
        // Matrix operations
        .def("add", &Matrix::add, py::arg("other"), "Add another matrix")
        .def("subtract", &Matrix::subtract, py::arg("other"), "Subtract another matrix")
        .def("multiply", &Matrix::multiply, py::arg("other"), "Multiply by another matrix")
        .def("scalar", &Matrix::scalar, py::arg("s"), "Multiply by scalar")
        .def("transpose", &Matrix::transpose, "Transpose the matrix")
        .def("dot", &Matrix::dot, py::arg("other"), "Compute dot product with another matrix")
        
        // Operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def("__rmul__", [](const Matrix& m, double s) {
            return m * s;
        }, py::is_operator())
        .def(py::self % py::self)
        
        // Utility methods
        .def("to_numpy", &matrix_to_numpy, "Convert matrix to numpy array")
        .def("__repr__", [](const Matrix& m) {
            std::ostringstream oss;
            oss << "<Matrix shape=(" << m.rows() << ", " << m.cols() << ")>";
            return oss.str();
        })
        .def("__str__", [](const Matrix& m) {
            return m.to_string(6);
        })
        .def_static("random_int", &Matrix::random_int,
                   py::arg("rows"), py::arg("cols"), py::arg("max_value") = 100,
                   "Create a matrix with random integer values");
    
    // Backend creation functions
    m.def("create_cpu_backend", &create_cpu_backend,
          "Create a CPU backend");
    
    #ifdef LUMIN_ENABLE_OPENMP
    m.def("create_omp_backend", &create_omp_backend,
          "Create an OpenMP backend");
    #endif
    
    #ifdef LUMIN_ENABLE_CUDA
    m.def("create_cuda_backend", &create_cuda_backend,
          "Create a CUDA backend");
    #endif
    
    #ifdef LUMIN_ENABLE_MPI
    m.def("create_mpi_backend", [](int comm) {
        // Note: MPI_Comm is typically an integer handle
        // This is a simplified version - you may need to adjust based on your MPI setup
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        if (comm != 0) {
            // For now, only support MPI_COMM_WORLD
            throw std::runtime_error("Only MPI_COMM_WORLD (0) is supported");
        }
        return create_mpi_backend(mpi_comm);
    }, py::arg("comm") = 0, "Create an MPI backend");
    #endif
    
    // Backend management
    m.def("set_default_backend", &set_default_backend,
          py::arg("backend"), "Set the default backend for matrix operations");
    
    m.def("get_default_backend", &get_default_backend,
          "Get the current default backend");
    
    // Convenience function to set backend by name
    m.def("set_backend", [](const std::string& name) {
        std::shared_ptr<Backend> backend;
        if (name == "cpu") {
            backend = create_cpu_backend();
        }
        #ifdef LUMIN_ENABLE_OPENMP
        else if (name == "openmp" || name == "omp") {
            backend = create_omp_backend();
        }
        #endif
        #ifdef LUMIN_ENABLE_CUDA
        else if (name == "cuda") {
            backend = create_cuda_backend();
        }
        #endif
        #ifdef LUMIN_ENABLE_MPI
        else if (name == "mpi") {
            backend = create_mpi_backend(MPI_COMM_WORLD);
        }
        #endif
        else {
            throw std::runtime_error("Unknown backend: " + name);
        }
        set_default_backend(backend);
    }, py::arg("name"), "Set backend by name (cpu, openmp, cuda, mpi)");
}
