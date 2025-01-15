#include "tensor.h"


Tensor::Tensor(std::vector<int64_t> shape) : shape(shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    data.resize(size, 0.0f); // Инициализируем тензор нулями
}

float& Tensor::operator[](std::vector<int64_t> indices) {
    int64_t index = 0;
    int64_t stride = 1;

    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        assert(indices[i] < shape[i]);
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return data[index];
}

const float& Tensor::operator[](std::vector<int64_t> indices) const {
    int64_t index = 0;
    int64_t stride = 1;

    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        assert(indices[i] < shape[i]);
        index += indices[i] * stride;
        stride *= shape[i];
    }
    return data[index];
}

void Tensor::set_value(std::vector<int64_t> indices, float value) {
    int64_t index = 0;
    int64_t stride = 1;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * stride;
        stride *= this->shape[i];  // Увеличиваем шаг для следующего измерения
    }
    this->data[index] = value;
}

void Tensor::print() const {
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << " ";
        if ((i + 1) % shape[1] == 0) std::cout << std::endl;
    }
}

Tensor Tensor::add(const Tensor& other) {
    assert(this->shape == other.shape && "Тензоры должны иметь одинаковую форму");
    Tensor result(this->shape);
    for (int64_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& other) {
    assert(this->shape == other.shape && "Тензоры должны иметь одинаковую форму");
    Tensor result(this->shape);
    for (int64_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] * other.data[i];
    }
    return result;
}