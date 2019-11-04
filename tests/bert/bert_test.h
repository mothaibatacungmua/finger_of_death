#include "bert_config.h"
#include "modeling.h"
#include "gtest/gtest.h"


namespace BERT
{
class BertTest : public ::testing::Test{
protected:
    MapAnyType config_map;
    BertConfig test_config;
    int batch_size = 10;
    int seq_length = 64;
    BertTest();

    virtual ~BertTest();

    virtual void SetUp();

    virtual void TearDown();
};
};