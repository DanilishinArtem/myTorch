#include "tensor.h"



int main(){
    // std::vector<int64_t> dim = {4,4};
    std::vector<std::vector<float>> elements = {{1,2,3},{4,5,6}};

    // Tensor tensor(dim);

    Tensor tensor(elements);

    tensor.print();
    return 0;
}