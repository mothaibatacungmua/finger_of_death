#include <map>
#include <memory>
#include <string>
#include "nlohmann/json.hpp"
#include "utils.h"


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

        static BertConfig fromMap(
            std::map<std::string, std::unique_ptr<MapFieldInterface>> mapObject);
        static BertConfig fromJson(nlohmann::json jsonObject);

        std::map<std::string, std::unique_ptr<MapFieldInterface>> toMap();
        nlohmann::json toJson();  
    };
}

