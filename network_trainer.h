#ifndef NETWORK_TRAINER_H
#define NETWORK_TRAINER_H

#include <vector>
#include <functional>
#include "training_data.h"
#include "neurons_network.h"

class NetworkTrainer
{
public:
    NetworkTrainer();

    void train_network(NeuronsNetwork &network, const std::vector<TrainingData> &datas, const float &epsilon, const int &nb_iterations);

    std::function<float(std::vector<float>, std::vector<float>)> m_cost = [](std::vector<float> a, std::vector<float> b) {
        int len = a.size();
        float diff(0);
        float cost(0);
        for (int i=0; i<len; i++) {
            diff = a[i]-b[i];
            cost += diff*diff;
        }
        return cost;
    };

private:
    void train_network_with_data(NeuronsNetwork &network, const TrainingData &datas, const float &epsilon);
};

#endif // NETWORK_TRAINER_H
