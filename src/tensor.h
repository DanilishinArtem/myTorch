#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int64_t> shape;

    Tensor(std::vector<int64_t> shape);

    int64_t compute_index(const std::vector<int64_t>& indices) const;

    float& at(const std::vector<int64_t>& indices);

    const float& at(const std::vector<int64_t>& indices) const;

    void set_value(std::vector<int64_t> indices, float value);

    void print() const;

    Tensor add(const Tensor& other);

    Tensor multiply(const Tensor& other);
};


#endif