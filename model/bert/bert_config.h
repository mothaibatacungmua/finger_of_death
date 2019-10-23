#include <map>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

using namespace std;


class MapFieldInterface
{
    ~MapFieldInterface() = default;
};


template <typename T>
class MapField : public MapFieldInterface {
    T _value;
public:
    T get(){ return this->_value; }
    MapField(T _value){
        this->_value = _value;
    };
};

#define GET_V(mapObject, type, field) (static_cast<MapField<type>*>(mapObject[field].get()))->get()
#define MAKE_V(type, value) unique_ptr<MapFieldInterface>(new MapField<type>(value))

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

    static BertConfig fromMap(map<string, unique_ptr<MapFieldInterface>> mapObject);
    static BertConfig fromJson(nlohmann::json jsonObject);

    map<string, unique_ptr<MapFieldInterface>> toMap();
    nlohmann::json toJson();  
};