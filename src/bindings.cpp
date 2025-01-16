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
            for(size_t i = 0; i < indices.size(); i++){
                idx.push_back(indices[i].cast<int64_t>());
            }
            return tensor.at(idx);
        })
        .def("__setitem__", &Tensor::set_value);
    m.def("create_tensor", [](std::vector<int64_t> shape) {
        return Tensor(shape);  // Функция создания Tensor
    });
}