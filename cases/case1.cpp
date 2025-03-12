#include "case1.h"

#include "../neurons_network/neurons_network.h"
#include "../neurons_network/neuronsnetworkfactory.h"
#include "../network_trainer.h"
#include "../training_data.h"

Case1::Case1() {}

std::vector<TrainingData> datas = {
    {{1,0,0,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0,0,0}},
    {{0,1,0,0,0,0,0,0,0,0}, {0,1,0,0,0,0,0,0,0,0}},
    {{0,0,1,0,0,0,0,0,0,0}, {0,0,1,0,0,0,0,0,0,0}},
    {{0,0,0,1,0,0,0,0,0,0}, {0,0,0,1,0,0,0,0,0,0}},
    {{0,0,0,0,1,0,0,0,0,0}, {0,0,0,0,1,0,0,0,0,0}},
    {{0,0,0,0,0,1,0,0,0,0}, {0,0,0,0,0,1,0,0,0,0}},
    {{0,0,0,0,0,0,1,0,0,0}, {0,0,0,0,0,0,1,0,0,0}},
    {{0,0,0,0,0,0,0,1,0,0}, {0,0,0,0,0,0,0,1,0,0}},
    {{0,0,0,0,0,0,0,0,1,0}, {0,0,0,0,0,0,0,0,1,0}},
    {{0,0,0,0,0,0,0,0,0,1}, {0,0,0,0,0,0,0,0,0,1}}
};


std::vector<float> map_data_to_input(const std::array<bool, 10> &data_array) {
    std::vector<float> res;
    res.reserve(10);
    for (bool data : data_array){
        res.emplace_back(data ? 1.f : 0.f);
    }
    return res;
}

void Case1::run() {
    NeuronsNetwork* network = NeuronsNetworkFactory::createNetwork(10, 10, 10, 2);
    NetworkTrainer network_trainer;
    std::vector<std::vector<TrainingData>> datasArray = {datas};
    network_trainer.train_network(*network, datasArray, datas, {0.1, 1000, 1});
    int res = network_trainer.test_network(*network, datas);
    std::cout << "RES: "<< res << "/" << datas.size() << std::flush;
}
