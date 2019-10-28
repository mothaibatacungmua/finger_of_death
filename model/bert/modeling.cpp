#include "modeling.h"
#include "common.h"

namespace BERT{
namespace Modeling{
    EmbeddingImpl::EmbeddingImpl(BertConfig config){
        word_embeddings = torch::nn::Embedding(
                            torch::nn::EmbeddingOptions(
                                config.vocab_size, config.hidden_size).padding_idx(0));
        position_embeddings = torch::nn::Embedding(
                            torch::nn::EmbeddingOptions(
                                config.max_position_embeddings, config.hidden_size));
        token_type_embeddings = torch::nn::Embedding(
                            torch::nn::EmbeddingOptions(
                                config.type_vocab_size, config.hidden_size));
        layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                std::vector<int64_t>{config.hidden_size}).eps(config.layer_norm_eps));
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions(config.hidden_dropout_prob));
    }

    torch::Tensor EmbeddingImpl::forward(torch::Tensor input_ids, 
             torch::Tensor position_ids, torch::Tensor token_type_ids){
        int seq_length = input_ids.size(1);
        if(!position_ids.defined()){
            position_ids = torch::arange(seq_length, 
                    torch::TensorOptions(torch::kLong));
            position_ids.to(input_ids.device());
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids);
        }

        if(!token_type_ids.defined()){
            token_type_ids = torch::zeros_like(input_ids);
        }

        auto word_embeddings = this->word_embeddings(input_ids);
        auto position_embeddings = this->position_embeddings(position_ids);
        auto token_type_embeddings = this->token_type_embeddings(token_type_ids);

        auto embeddings = word_embeddings + position_embeddings + token_type_embeddings;

        embeddings = layer_norm(embeddings);
        embeddings = dropout(embeddings);

        return embeddings;
    }

    SelfAttentionImpl::SelfAttentionImpl(BertConfig config){
        if (config.hidden_size % config.num_attention_heads != 0){
            throw std::runtime_error(string_format(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)", 
                config.hidden_size, config.num_attention_heads));
        }
        output_attentions = config.output_attentions;

        num_attention_heads = config.num_attention_heads;
        attention_head_size = (int)(config.hidden_size / config.num_attention_heads);
        all_head_size = num_attention_heads * attention_head_size;

        query = torch::nn::Linear(config.hidden_size, all_head_size);
        key = torch::nn::Linear(config.hidden_size, all_head_size);
        value = torch::nn::Linear(config.hidden_size, all_head_size);
    }

    torch::Tensor SelfAttentionImpl::TransposeForScores(torch::Tensor x){
        /* 
            [batch_size, seq_length, hidden_size] 
                -> [batch_size, seq_length, num_attention_heads, attention_head_size]
                -> [batch_size, num_attention_heads, seq_length, attention_head_size]
        */
        auto new_x_shape = c10::IntArrayRef(x.sizes().begin(), x.sizes().end()-1).vec();
        new_x_shape.push_back(num_attention_heads);
        new_x_shape.push_back(attention_head_size);
        
        x = x.view(new_x_shape);
        return x.permute({0, 2, 1, 3});
    }

    std::vector<torch::Tensor> SelfAttentionImpl::forward(
        torch::Tensor hidden_states,
        torch::Tensor attention_mask,
        torch::Tensor head_mask
    ){
        /** Return shape [batch_size, seq_length, hidden_size] **/
        auto mixed_query_layer = this->query(hidden_states);
        auto mixed_key_layer = this->key(hidden_states);
        auto mixed_value_layer = this->value(hidden_states);

        /** Return shape [batch_size, num_attention_heads, seq_length, attention_head_size] **/
        auto query_layer = this->TransposeForScores(mixed_query_layer);
        auto key_layer = this->TransposeForScores(mixed_key_layer);
        auto value_layer = this->TransposeForScores(mixed_value_layer);

        /** [batch_size, num_attention_heads, seq_length, attention_head_size] x
         *  [batch_size, num_attention_heads, attention_head_size, seq_length] x
         * = [batch_size, num_attention_heads, seq_length, seq_length]
         */ 
        auto attention_scores = torch::matmul(query_layer, key_layer.transpose(-1, -2));

        if(!attention_mask.defined()){
            attention_scores = attention_scores + attention_mask;
        }

        /** attention_probs has shape [batch_size, num_attention_heads, seq_length, seq_length] **/
        auto attention_probs = torch::nn::Softmax(torch::nn::SoftmaxOptions(-1))(attention_scores);

        attention_probs = dropout(attention_probs);

        if(!head_mask.defined()){
            attention_probs = attention_probs * head_mask;
        }

        auto context_layer = torch::matmul(attention_probs, value_layer);

        /** attention_probs has shape [batch_size, seq_length, num_attention_heads, seq_length] **/
        context_layer = context_layer.permute({0, 2, 1, 3}).contiguous();

        auto new_cl_shape = c10::IntArrayRef(
            context_layer.sizes().begin(), context_layer.sizes().end()-2).vec();
        new_cl_shape.push_back(all_head_size);
        context_layer = context_layer.view(new_cl_shape);

        std::vector<torch::Tensor> output;

        output.push_back(context_layer);
        if(output_attentions){
            output.push_back(attention_probs);
        }
        
        return output;
    }
}
}
