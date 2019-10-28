#include "bert_config.h"
#include <iostream>


int main(){
    BERT::BertConfig config;

    std::cout << config.vocab_size;

    return 0;
}