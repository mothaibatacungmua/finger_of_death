#include "bert_config.h"
#include <torch/torch.h>

// https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py

namespace BERT{
namespace Modeling{
    struct EmbeddingImpl : torch::nn::Module{
        EmbeddingImpl(BertConfig config);

        torch::nn::Embedding word_embeddings;
        torch::nn::Embedding position_embeddings;
        torch::nn::Embedding token_type_embeddings;

        torch::nn::LayerNorm layer_norm;
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

        torch::nn::Linear query;
        torch::nn::Linear key;
        torch::nn::Linear value;
        torch::nn::Dropout dropout;

        torch::Tensor TransposeForScores(torch::Tensor x);
        torch::Tensor forward(
            torch::Tensor hidden_states, 
            torch::Tensor attention_mask={},
            torch::Tensor head_mask={});
    };
    TORCH_MODULE(SelfAttention);
}
}