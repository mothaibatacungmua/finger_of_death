#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace std;
// https://pytorch.org/cppdocs/api/namespace_torch__nn.html

struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M) 
        : linear(register_module("linear", torch::nn::Linear(N, M))){
        another_bias = register_parameter("b", torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input){
        return linear(input) + another_bias;
    }
    torch::nn::Linear linear;
    torch::Tensor another_bias;
};

string get_name(){
    return "Alex";
}

void print_ref_1(const string& name){
    cout << name;
}

void print_ref_2(string&& name){
    cout << name;
}

int main(){
    // Net net(4, 5);
    // for(const auto& p: net.parameters()){
    //     std::cout << p << std::endl;
    // }

    string me("Alex");
    print_ref_2(me);
}