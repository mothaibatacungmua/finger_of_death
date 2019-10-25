#include "bert_config.h"
#include <iostream>


int main(){
    BERT::BertConfig config;

    std::cout << config.vocabSize;

    return 0;
}