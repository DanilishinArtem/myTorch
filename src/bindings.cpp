#include <pybind11/pybind11.h>
#include "tensor.h"  // подключаем наш C++ код

namespace py = pybind11;

PYBIND11_MODULE(libtorchcpp, m) {
    py::class_<Tensor>(m, "Tensor")
        .def_readwrite("data", &Tensor::data)
        .def(py::init<std::vector<int64_t>>())
        .def("add", &Tensor::add)
        .def("multiply", &Tensor::multiply)
        .def("print", &Tensor::print)
        .def("__getitem__", [](const Tensor& tensor, py::tuple indices) -> float {
            std::vector<int64_t> idx;
            for (auto item : indices) {
                idx.push_back(item.cast<int64_t>());
            }
            return tensor.at(idx);
        })
        .def("__setitem__", [](const Tensor& tensor, py::tuple indices, float value) {
            std::vector<int64_t> idx;
            for (auto item : indices) {
                idx.push_back(item.cast<int64_t>());
            }
            tensor.at(idx) = value;
        });
    m.def("create_tensor", [](std::vector<int64_t> shape) {
        return Tensor(shape);  // Функция создания Tensor
    });
}