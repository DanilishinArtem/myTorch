#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int64_t> shape;

    Tensor(std::vector<int64_t> shape);

    template <typename T>
    Tensor(const T& nested_data);

    const float& at(const std::vector<int64_t>& indices) const;

    float& at(const std::vector<int64_t>& indices);

    void print() const;

    void set_value(std::vector<int64_t> indices, float value);

    Tensor add(const Tensor& other);

    Tensor multiply(const Tensor& other);

    template <typename T>
    std::vector<int64_t> infer_shape(const T& vec);

    int64_t compute_index(const std::vector<int64_t>& indices) const;

    int64_t compute_total_size(const std::vector<int64_t>& dims) const;

    template <typename T1, typename T2>
    void flatten_data(const T1& nested_data, std::vector<T2>& flat_data);
};

inline Tensor::Tensor(std::vector<int64_t> shape) : shape(shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    data.resize(size, 0.0f); // Инициализируем тензор нулями
}

template <typename T>
inline Tensor::Tensor(const T& nested_data) {
    shape = infer_shape(nested_data);
    data.reserve(compute_total_size(shape));
    flatten_data(nested_data, data);
    // std::cout << "hello world" << std::endl;
}

template <typename T>
std::vector<int64_t> Tensor::infer_shape(const T& vec) {
    std::vector<int64_t> dims{static_cast<int64_t>(vec.size())};
    if constexpr (std::is_arithmetic_v<typename T::value_type>) {
        return dims;
    } else {
        if (!vec.empty()) {
            auto sub_shape = infer_shape(vec[0]);
            dims.insert(dims.end(), sub_shape.begin(), sub_shape.end());
        }
    }
    return dims;
}

template <typename T1, typename T2>
void Tensor::flatten_data(const T1& nested_data, std::vector<T2>& flat_data){
    if constexpr (std::is_arithmetic_v<T1>) {
        // Если T — число (int, float, double и т.д.)
        flat_data.push_back(static_cast<T2>(nested_data));
    } else if constexpr (std::is_same_v<T1, std::vector<typename T1::value_type>>) {
        // Если T — вектор
        for (const auto& item : nested_data) {
            flatten_data(item, flat_data);
        }
    } else {
        throw std::invalid_argument("Incorrect data format.");
    }
}

#endif