cmake_minimum_required(VERSION 3.12)
project(torchcpp)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

pybind11_add_module(libtorchcpp src/bindings.cpp src/tensor.cpp)

target_link_libraries(libtorchcpp PRIVATE pybind11::module)
