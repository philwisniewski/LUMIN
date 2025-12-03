#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.hpp"
#include "factory.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lumin, m) {
  py::class_<lumin::Matrix>(m, "Matrix")
    .def(py::init<size_t,size_t>())
    .def("add", &lumin::Matrix::add)
    .def("multiply", &lumin::Matrix::multiply)
    .def("scalar", &lumin::Matrix::scalar)
    .def("transpose", &lumin::Matrix::transpose);

  m.def("set_backend", [](int t) {
    lumin::set_backend((lumin::BackendType) t);
  });
}
