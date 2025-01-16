#include "tensor.h"


Tensor::Tensor(std::vector<int64_t> shape) : shape(shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    data.resize(size, 0.0f); // Инициализируем тензор нулями
}

int64_t Tensor::compute_index(const std::vector<int64_t>& indices) const{
    if(indices.size() != this->shape.size()){
        throw std::out_of_range("Number of indices is not equal to the number of dimensions");
    }
    int64_t index = 0;
    int64_t stride = 1;
    for(size_t i = 0; i < indices.size(); ++i){
        if(indices[i] < 0 || indices[i] >= this->shape[i]){
            throw std::out_of_range("Index is out of range");
        }
        index += indices[i] * stride;
        stride *= this->shape[i];
    }
    return index;
}

float& Tensor::at(const std::vector<int64_t>& indices){
    return this->data[this->compute_index(indices)];
}

const float& Tensor::at(const std::vector<int64_t>& indices) const{
    return this->data[this->compute_index(indices)];
}

void Tensor::set_value(std::vector<int64_t> indices, float value) {
    this->data[this->compute_index(indices)] = value;
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