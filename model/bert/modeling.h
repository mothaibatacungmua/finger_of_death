#ifndef BERT_MODELING_H_
#define BERT_MODELING_H_

#include "bert_config.h"
#include <torch/torch.h>

// https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py

namespace BERT{
namespace Modeling{
    struct EmbeddingImpl : torch::nn::Module{
        EmbeddingImpl(BertConfig config);

        torch::nn::Embedding word_embeddings = nullptr;
        torch::nn::Embedding position_embeddings = nullptr;
        torch::nn::Embedding token_type_embeddings = nullptr;

        torch::nn::LayerNorm layer_norm = nullptr;
        torch::nn::Dropout dropout;

        torch::Tensor forward(torch::Tensor input_ids, 
            torch::Tensor position_ids, torch::Tensor token_type_ids);
    };
    TORCH_MODULE(Embedding);


    struct SelfAttentionImpl : torch::nn::Module{
        SelfAttentionImpl(BertConfig config);

        bool output_attentions;
        int num_attention_heads;
        int attention_head_size;
        int all_head_size;

        torch::nn::Linear query = nullptr;
        torch::nn::Linear key = nullptr;
        torch::nn::Linear value = nullptr;
        torch::nn::Dropout dropout;

        torch::Tensor TransposeForScores(torch::Tensor x);
        std::vector<torch::Tensor> forward(
            torch::Tensor hidden_states, 
            torch::Tensor attention_mask={},
            torch::Tensor head_mask={},
            torch::Tensor encoder_hidden_states={}, 
            torch::Tensor encoder_attention_mask={});
    };
    TORCH_MODULE(SelfAttention);

    struct SelfOutputImpl : torch::nn::Module{
        SelfOutputImpl(BertConfig config);

        torch::nn::Linear dense = nullptr;
        torch::nn::LayerNorm layer_norm = nullptr;
        torch::nn::Dropout dropout;

        torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor input_tensor);
    };
    TORCH_MODULE(SelfOutput);

    struct BertAttentionImpl : torch::nn::Module{
        BertAttentionImpl(BertConfig config);

        SelfAttention self;
        SelfOutput output;
        std::set<int> pruned_heads;

        std::vector<torch::Tensor> forward(
            torch::Tensor hidden_states,
            torch::Tensor attention_mask={},
            torch::Tensor head_mask={},
            torch::Tensor encoder_hidden_states={}, 
            torch::Tensor encoder_attention_mask={}
        );

        void prunded_heads(std::vector<int> heads);
    };
    TORCH_MODULE(BertAttention);

}
}

#endif