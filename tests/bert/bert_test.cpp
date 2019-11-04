#include <fstream>
#include "bert_test.h"

namespace BERT
{
BertTest::BertTest() {}

BertTest::~BertTest() {};

void BertTest::SetUp() {
    config_map.insert(MAKE_P("vocab_size", int, 6897));
    config_map.insert(MAKE_P("hidden_size", int, 128));
    config_map.insert(MAKE_P("num_hidden_layers", int, 4));
    config_map.insert(MAKE_P("num_attention_heads", int, 4));
    config_map.insert(MAKE_P("intermediate_size", int, 128));
    config_map.insert(MAKE_P("hidden_act", std::string, "gelu"));
    config_map.insert(MAKE_P("hidden_dropout_prob", float, 0.1f));
    config_map.insert(MAKE_P("attention_probs_dropout_prob", float, 0.1f));
    config_map.insert(MAKE_P("max_position_embeddings", int, 256));
    config_map.insert(MAKE_P("type_vocab_size", int, 2));
    config_map.insert(MAKE_P("initializer_range", float, 0.01f));

    test_config = BERT::BertConfig::fromMap(config_map);
};

void BertTest::TearDown() {};

TEST_F(BertTest, TestFromMap) {
    BERT::BertConfig config = BERT::BertConfig::fromMap(config_map);

    EXPECT_EQ(config.vocab_size, 6897);
    EXPECT_EQ(config.hidden_size, 128);
    EXPECT_EQ(config.num_hidden_layers, 4);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.intermediate_size, 128);
    EXPECT_EQ(config.hidden_act.compare("gelu"), 0);
    EXPECT_EQ(config.hidden_dropout_prob, 0.1f);
    EXPECT_EQ(config.attention_probs_dropout_prob, 0.1f);
    EXPECT_EQ(config.max_position_embeddings, 256);
    EXPECT_EQ(config.type_vocab_size, 2);
    EXPECT_EQ(config.initializer_range, 0.01f);
}

TEST_F(BertTest, TestFromJson){
    std::ifstream i("data/bert_config.json");
    
    nlohmann::json cj;
    i >> cj;

    BertConfig config = BertConfig::fromJson(cj);

    EXPECT_EQ(config.vocab_size, 6897);
    EXPECT_EQ(config.hidden_size, 128);
    EXPECT_EQ(config.num_hidden_layers, 4);
    EXPECT_EQ(config.num_attention_heads, 4);
    EXPECT_EQ(config.intermediate_size, 128);
    EXPECT_EQ(config.hidden_act.compare("gelu"), 0);
    EXPECT_EQ(config.hidden_dropout_prob, 0.1f);
    EXPECT_EQ(config.attention_probs_dropout_prob, 0.1f);
    EXPECT_EQ(config.max_position_embeddings, 256);
    EXPECT_EQ(config.type_vocab_size, 2);
    EXPECT_EQ(config.initializer_range, 0.01f);
}

// TEST_F(BertTest, TestEmbeddingShape){
//     Modeling::Embedding embed_layer(config_map);

//     torch::Tensor input_ids = torch::randint(0,)
//     std::cout << embed_layer.forward() << std::endl;
// }
}