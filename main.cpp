#include <iostream>
#include <map>
#include <vector>
#include <array>

#include "neurons_network.h"
#include "network_trainer.h"
#include "training_data.h"
#include "vector.h"

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


int main()
{
    // Vector<float> test({1,0,0,0,2,0,0,0,0,0});
    // Vector<float> test2({1,0,0,0,1,0,0,0,0,0});
    // test[0] = 6;
    // std::cout << (test.dot(test2)) << std::endl;
    // std::cout << (test * 5)[0];
    NeuronsNetwork network;
    NetworkTrainer network_trainer;

    network_trainer.train_network(network, datas, 0.1, 1000);

    auto res = network.compute(map_data_to_input({0,0,0,0,1,0,0,0,0,0}));
    std::cout << res;
    return 0;
}
