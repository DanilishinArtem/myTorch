#include <pybind11/pybind11.h>
#include "tensor.h"  // подключаем наш C++ код

namespace py = pybind11;

PYBIND11_MODULE(libtorchcpp, m) {
    py::class_<Tensor>(m, "Tensor")
        .def_readwrite("data", &Tensor::data)
        .def(py::init<std::vector<int64_t>>())
        .def("add", &Tensor::add)
        .def("multiply", &Tensor::multiply)
        .def("__setitem__", &Tensor::set_value)
        .def("print", &Tensor::print)
        .def("__getitem__", (const float& (Tensor::*)(std::vector<int64_t>) const) &Tensor::operator[], py::arg("indices"))
        .def("__getitem__", (float& (Tensor::*)(std::vector<int64_t>)) &Tensor::operator[], py::arg("indices"));
    m.def("create_tensor", [](std::vector<int64_t> shape) {
        return Tensor(shape);  // Функция создания Tensor
    });
}