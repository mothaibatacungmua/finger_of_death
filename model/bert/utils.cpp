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

torch::nn::Linear PruneLinearLayer(
    torch::nn::Linear layer, torch::Tensor index, int dim=0){
    // Prune a linear layer (a model parameters) to keep only entries in index.
    // Return the pruned layer as a new layer with requires_grad=True
    // Used to remove heads
    index = index.to((layer->weight).device());
    auto W = layer->weight.index_select(dim, index).clone().detach();
    torch::Tensor b = {};

    if(layer->bias.defined()){
        if(dim == 1){
            b = layer->bias.clone().detach();
        }else{
            b = layer->bias[index].clone().detach();
        }
    }

    auto new_size = layer->weight.sizes().vec();
    new_size[dim] = index.sizes()[0];
    auto new_layer = torch::nn::Linear(
                        torch::nn::LinearOptions(new_size[1], new_size[0])
                        .bias(layer->bias.defined()));
    new_layer->weight.requires_grad = false;
    new_layer->weight.copy_(W.contiguous());
    new_layer->weight.requires_grad = true;

    if(layer->bias.defined()){
        new_layer->bias.requires_grad = false;
        new_layer->bias.copy_(b.contiguous());
        new_layer->bias.requires_grad = true;
    }
    return new_layer;
}