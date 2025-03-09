#include "case1.h"

#include "../neurons_network.h"
#include "../neuronsnetworkfactory.h"
#include "../network_trainer.h"
#include "../training_data.h"
#include "../vector.h"

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
    network_trainer.train_network(*network, datas, 0.1, 1000);

    auto res = network->compute(map_data_to_input({0,0,0,0,1,0,0,0,0,0}));
    std::cout << "RES: "<< res << std::flush;
}
