#include <torch/torch.h>


namespace BERT
{   
    using ActFn = torch::Tensor(*)(torch::Tensor);
    class MapFieldInterface
    {
    public:
        ~MapFieldInterface() = default;
    };


    template <typename T>
    class MapField : public MapFieldInterface {
        T _value;
    public:
        T get(){ return this->_value; }
        MapField(T _value){
            this->_value = _value;
        };
    };

    torch::Tensor Gelu(torch::Tensor x);
    torch::Tensor Swish(torch::Tensor x);
    torch::Tensor Relu(torch::Tensor x);

    ActFn GetActFn(const std::string name="gelu");
}

#define GET_V(mapObject, type, field) (static_cast<BERT::MapField<type>*>(mapObject[field].get()))->get()
#define MAKE_V(type, value) std::unique_ptr<BERT::MapFieldInterface>(new BERT::MapField<type>(value))
