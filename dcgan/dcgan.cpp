#include <torch/torch.h>
#include <iostream>
#include <cstdio>
#include <cmath>

using namespace torch;
// https://pytorch.org/cppdocs/api/namespace_torch__nn.html


struct GeneratorImpl : nn::Module {
    GeneratorImpl (int kNoiseSize)
        : conv1(nn::Conv2dOptions(kNoiseSize, 256, 4)
                    .with_bias(false)
                    .transposed(true)),
          batch_norm1(256),
          conv2(nn::Conv2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
          batch_norm2(128),
          conv3(nn::Conv2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true)),
          batch_norm3(64),
          conv4(nn::Conv2dOptions(64, 1, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)
                    .transposed(true))

    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::relu(conv4(x));
        return x;
    }

    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(Generator);


struct DiscriminatorImpl : nn::Module {
    DiscriminatorImpl()
        : conv1(nn::Conv2dOptions(1, 64, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
          batch_norm1(64),
          conv2(nn::Conv2dOptions(64, 128, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
          batch_norm2(128),
          conv3(nn::Conv2dOptions(128, 256, 4)
                    .stride(2)
                    .padding(1)
                    .with_bias(false)),
          batch_norm3(256),
          conv4(nn::Conv2dOptions(256, 1, 3)
                    .stride(1)
                    .padding(0)
                    .with_bias(false))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    };

    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm batch_norm1, batch_norm2, batch_norm3;
        
};
TORCH_MODULE(Discriminator);


int main(){
    return 0;
}