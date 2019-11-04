#ifndef BERT_CONFIG_H_
#define BERT_CONFIG_H_

#include <map>
#include <memory>
#include <string>
#include "nlohmann/json.hpp"
#include "utils.h"

#define MapAnyType std::map<std::string, std::unique_ptr<BERT::MapFieldInterface>>
#define NUMBER_OF_FIELDS 11

namespace BERT
{
    class BertConfig
    {
        
    public:
        int vocab_size;
        int hidden_size;
        int num_hidden_layers;
        int num_attention_heads;
        int intermediate_size;
        std::string hidden_act;
        float hidden_dropout_prob;
        float attention_probs_dropout_prob;
        float max_position_embeddings;
        int type_vocab_size;
        float initializer_range;
        float layer_norm_eps;
        bool output_attentions;

        // Construction with full parameters
        BertConfig(
            int vocab_size,
            int hidden_size,
            int num_hidden_layers,
            int num_attention_heads,
            int intermediate_size,
            std::string hidden_act,
            float hidden_dropout_prob,
            float attention_probs_dropout_prob,
            float max_position_embeddings,
            int type_vocab_size,
            float initializer_range
        );

        // Construction without any parameters
        BertConfig();
        ~BertConfig(){};

        static std::string _fields[NUMBER_OF_FIELDS];
        static BertConfig fromMap(MapAnyType& mapObject);
        static BertConfig fromJson(nlohmann::json jsonObject);

        MapAnyType toMap();
        nlohmann::json toJson();  

        BertConfig& operator=(BertConfig const &obj);
    };
}

#endif