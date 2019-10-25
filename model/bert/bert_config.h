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
        int vocabSize;
        int hiddenSize;
        int numHiddenLayers;
        int numAttentionHeads;
        int intermediateSize;
        std::string hiddenAct;
        float hiddenDropoutProb;
        float attentionProbsDropoutProb;
        float maxPositionEmbeddings;
        int typeVocabSize;
        float initializerRange;

        // Construction with full parameters
        BertConfig(
            int vocabSize,
            int hiddenSize,
            int numHiddenLayers,
            int numAttentionHeads,
            int intermediateSize,
            std::string hiddenAct,
            float hiddenDropoutProb,
            float attentionProbsDropoutProb,
            float maxPositionEmbeddings,
            int typeVocabSize,
            float initializerRange
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

