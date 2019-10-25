#include "utils.h"
#include <math.h>


torch::Tensor BERT::Gelu(torch::Tensor x){
    torch::Tensor cdf = 0.5*(1 + torch::tanh(sqrt(2. / M_PI) * (x + 0.044715 * torch::pow(x, 3))));

    return x * cdf;
}

torch::Tensor BERT::Swish(torch::Tensor x){
    return x * torch::sigmoid(x);
}

torch::Tensor BERT::Relu(torch::Tensor x){
    return torch::relu(x);
}

BERT::ActFn BERT::GetActFn(std::string name){
    if(name.compare("gelu") == 0){
        return &BERT::Gelu;
    }else if(name.compare("relu") == 0){
        return &BERT::Relu;
    }else if(name.compare("swish") == 0){
        return &BERT::Swish;
    }

    throw std::runtime_error(
        std::string("Not found activation`") + name + std::string("`function"));
}