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

    Tensor(std::initializer_list<std::initializer_list<float>> nested_data);

    const float& at(const std::vector<int64_t>& indices) const;

    float& at(const std::vector<int64_t>& indices);

    void print() const;

    void set_value(std::vector<int64_t> indices, float value);

    Tensor add(const Tensor& other);

    Tensor multiply(const Tensor& other);

// private:
    std::vector<int64_t> infer_shape(const std::initializer_list<std::initializer_list<float>>& nested_data);

    int64_t compute_index(const std::vector<int64_t>& indices) const;

    int64_t compute_total_size(const std::vector<int64_t>& dims) const;

    void flatten_data(const std::initializer_list<std::initializer_list<float>>& nested_data, std::vector<float>& flat_data);
};


#endif