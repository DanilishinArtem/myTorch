#include "tensor.h"


Tensor::Tensor(std::vector<int64_t> shape) : shape(shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    data.resize(size, 0.0f); // Инициализируем тензор нулями
}

Tensor::Tensor(std::initializer_list<std::any> nested_data) : shape(shape) {
    shape = infer_shape(nested_data);
    data.reserve(compute_total_size(shape));
    flatten_data(nested_data, data);
}

float& Tensor::at(const std::vector<int64_t>& indices){
    return this->data[this->compute_index(indices)];
}

const float& Tensor::at(const std::vector<int64_t>& indices) const{
    return this->data[this->compute_index(indices)];
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

int64_t Tensor::compute_total_size(const std::vector<int64_t>& dims) const {
    int64_t size = 1;
    for (int64_t dim : dims) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> Tensor::infer_shape(const std::any& nested_data) {
    if (nested_data.type() == typeid(std::initializer_list<std::any>)) {
        auto list = std::any_cast<std::initializer_list<std::any>>(nested_data);
        std::vector<int64_t> dims{static_cast<int64_t>(list.size())};
        if (!list.empty()) {
            auto sub_shape = infer_shape(*list.begin());
            dims.insert(dims.end(), sub_shape.begin(), sub_shape.end());
        }
        return dims;
    }
    return {};
}

void flatten_data(const std::initializer_list<std::initializer_list<float>>& nested_data, std::vector<float>& flat_data){
    if (nested_data.type() == typeid(std::initializer_list<std::any>)) {
        auto list = std::any_cast<std::initializer_list<std::any>>(nested_data);
        for (const auto& item : list) {
            flatten_data(item, flat_data);
        }
    } else if (nested_data.type() == typeid(float)) {
        flat_data.push_back(std::any_cast<float>(nested_data));
    } else {
        throw std::invalid_argument("Некорректный формат данных.");
    }
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