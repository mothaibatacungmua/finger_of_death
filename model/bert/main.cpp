#include "bert_config.h"
#include <iostream>


int main(){
    MapAnyType config_map;
    config_map.insert(MAKE_P("vocab_size", int, 6897));
    config_map.insert(MAKE_P("hidden_size", int, 128));
    config_map.insert(MAKE_P("num_hidden_layers", int, 4));
    config_map.insert(MAKE_P("num_attention_heads", int, 4));
    config_map.insert(MAKE_P("intermediate_size", int, 128));
    config_map.insert(MAKE_P("hidden_act", std::string, "gelu"));
    config_map.insert(MAKE_P("hidden_dropout_prob", float, 0.1));
    config_map.insert(MAKE_P("attention_probs_dropout_prob", float, 0.1));
    config_map.insert(MAKE_P("max_position_embeddings", int, 256));
    config_map.insert(MAKE_P("type_vocab_size", int, 2));
    config_map.insert(MAKE_P("initializer_range", float, 0.01));

    auto config = BERT::BertConfig::fromMap(config_map);

    std::cout << config.hidden_size << "\n";
    return 0;
}