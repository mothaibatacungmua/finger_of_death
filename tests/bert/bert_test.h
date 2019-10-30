#include "gtest/gtest.h"

class BertTest : public ::testing::Test{
protected:
    BertTest();

    virtual ~BertTest();

    virtual void SetUp();

    virtual void TearDown();
};