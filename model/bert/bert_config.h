#include <map>
#include <memory>
#include <string>
using namespace std;

class BertConfig
{
public:
    int vocabSize;
    int hiddenSize;
    int numHiddenLayers;
    int numAttentionHeads;
    int intermediateSize;
    string hiddenAct;
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
        string hiddenAct,
        float hiddenDropoutProb,
        float attentionProbsDropoutProb,
        float maxPositionEmbeddings,
        int typeVocabSize,
        float initializerRange
    );

    // Construction without any parameters
    BertConfig();
    ~BertConfig();

    static BertConfig fromMap(map<string, unique_ptr<void>> mapObject);
    static BertConfig fromJson();

    map<string, unique_ptr<void>> toMap();
    void toJson();  
};