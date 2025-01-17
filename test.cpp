#include <iostream>
#include <vector>
#include <type_traits>
#include <stdexcept>


template <typename T>
std::vector<int64_t> infer_shape(const T& vec) {
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

int64_t compute_total_size(const std::vector<int64_t>& dims) {
    int64_t size = 1;
    for (int64_t dim : dims) {
        size *= dim;
    }
    return size;
}

template <typename T1, typename T2>
void flatten_data(const T1& nested_data, std::vector<T2>& flat_data) {
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

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& str) {
    out << "[";
    for (size_t i = 0; i < str.size(); i++) {
        if (i < str.size() - 1) {
            out << str[i] << ",";
        } else {
            out << str[i];
        }
    }
    out << "]";
    return out;
}

int main() {
    // Пример с вложенными векторами
    std::vector<std::vector<std::vector<float>>> nested_data = {
        {{1.1, 2, 3}, {4, 5, 6}}, 
        {{7, 8, 9}, {10, 11, 12}}
    };
    std::vector<int64_t> data;
    std::vector<int64_t> shape = infer_shape(nested_data);
    data.reserve(compute_total_size(shape));
    flatten_data(nested_data, data);
    std::cout << "Flattened data: " << data << std::endl;
    std::cout << "Shape: " << shape << std::endl;
    return 0;
}
