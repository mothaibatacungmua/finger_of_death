#include "bert_config.h"

namespace BERT
{
    BertConfig::BertConfig(int vocabSize,
            int hiddenSize,
            int numHiddenLayers,
            int numAttentionHeads,
            int intermediateSize,
            std::string hiddenAct,
            float hiddenDropoutProb,
            float attentionProbsDropoutProb,
            float maxPositionEmbeddings,
            int typeVocabSize,
            float initializerRange)
    {
        this->vocabSize = vocabSize;
        this->hiddenSize = hiddenSize;
        this->numHiddenLayers = numHiddenLayers;
        this->numAttentionHeads = numAttentionHeads;
        this->intermediateSize = intermediateSize;
        this->hiddenAct = hiddenAct;
        this->hiddenDropoutProb = hiddenDropoutProb;
        this->attentionProbsDropoutProb = attentionProbsDropoutProb;
        this->maxPositionEmbeddings = maxPositionEmbeddings;
        this->typeVocabSize = typeVocabSize;
        this->initializerRange = initializerRange;
    };


    BertConfig::BertConfig():hiddenSize(768),numHiddenLayers(12),
        numAttentionHeads(12), intermediateSize(3072),
        hiddenAct("gelu"), hiddenDropoutProb(0.1f),
        attentionProbsDropoutProb(0.1f), maxPositionEmbeddings(512),
        typeVocabSize(16), initializerRange(0.02){};


    BertConfig BertConfig::fromMap(
        std::map<std::string, std::unique_ptr<MapFieldInterface>> mapObject){
        BertConfig ret;

        ret.vocabSize = GET_V(mapObject, int, "vocabSize");
        ret.hiddenSize = GET_V(mapObject, int, "hiddenSize");
        ret.numHiddenLayers = GET_V(mapObject, int, "numHiddenLayers");
        ret.numAttentionHeads = GET_V(mapObject, int, "numAttentionHeads");
        ret.intermediateSize = GET_V(mapObject, int, "intermediateSize");
        ret.hiddenAct = GET_V(mapObject, std::string, "hiddenAct");
        ret.hiddenDropoutProb = GET_V(mapObject, float, "hiddenDropoutProb");
        ret.attentionProbsDropoutProb = GET_V(mapObject, float, "attentionProbsDropoutProb");
        ret.maxPositionEmbeddings = GET_V(mapObject, int, "maxPositionEmbeddings");
        ret.typeVocabSize = GET_V(mapObject, int, "typeVocabSize");
        ret.initializerRange = GET_V(mapObject, float, "initializerRange");

        return ret;
    }


    BertConfig BertConfig::fromJson(nlohmann::json jsonObject){
        BertConfig ret;

        ret.vocabSize = jsonObject["vocabSize"];
        ret.hiddenSize = jsonObject["hiddenSize"];
        ret.numHiddenLayers = jsonObject["numHiddenLayers"];
        ret.numAttentionHeads = jsonObject["numAttentionHeads"];
        ret.intermediateSize = jsonObject["intermediateSize"];
        ret.hiddenAct = jsonObject["hiddenAct"];
        ret.hiddenDropoutProb = jsonObject["hiddenDropoutProb"];
        ret.attentionProbsDropoutProb = jsonObject["attentionProbsDropoutProb"];
        ret.maxPositionEmbeddings = jsonObject["maxPositionEmbeddings"];
        ret.typeVocabSize = jsonObject["typeVocabSize"];
        ret.initializerRange = jsonObject["initializerRange"];

        return ret;
    }


    std::map<std::string, std::unique_ptr<MapFieldInterface>> BertConfig::toMap(){
        std::map<std::string, std::unique_ptr<MapFieldInterface>> ret;

        ret["vocabSize"] = MAKE_V(int, this->vocabSize);
        ret["hiddenSize"] = MAKE_V(int, this->hiddenSize);
        ret["numHiddenLayers"] = MAKE_V(int, this->numHiddenLayers);
        ret["numAttentionHeads"] = MAKE_V(int, this->numAttentionHeads);
        ret["intermediateSize"] = MAKE_V(int, this->intermediateSize);
        ret["hiddenAct"] = MAKE_V(std::string, this->hiddenAct);
        ret["hiddenDropoutProb"] = MAKE_V(float, this->hiddenDropoutProb);
        ret["attentionProbsDropoutProb"] = MAKE_V(float, this->attentionProbsDropoutProb);
        ret["maxPositionEmbeddings"] = MAKE_V(int, this->maxPositionEmbeddings);
        ret["typeVocabSize"] = MAKE_V(int, this->typeVocabSize);
        ret["initializerRange"] = MAKE_V(float, this->initializerRange);
    }


    nlohmann::json BertConfig::toJson(){
        nlohmann::json ret;

        ret["vocabSize"] = this->vocabSize;
        ret["hiddenSize"] = this->hiddenSize;
        ret["numHiddenLayers"] = this->numHiddenLayers;
        ret["numAttentionHeads"] = this->numAttentionHeads;
        ret["intermediateSize"] = this->intermediateSize;
        ret["hiddenAct"] = this->hiddenAct;
        ret["hiddenDropoutProb"] = this->hiddenDropoutProb;
        ret["attentionProbsDropoutProb"] = this->attentionProbsDropoutProb;
        ret["maxPositionEmbeddings"] = this->maxPositionEmbeddings;
        ret["typeVocabSize"] = this->typeVocabSize;
        ret["initializerRange"] = this->initializerRange;

        return ret;
    }
}