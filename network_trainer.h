#ifndef NETWORK_TRAINER_H
#define NETWORK_TRAINER_H

#include <vector>
#include "training_data.h"
#include "neurons_network.h"

class NetworkTrainer
{
public:
    NetworkTrainer();

    void train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, std::vector<TrainingData> test_datas,
                       const float &epsilon, const int &nb_epochs, const float &max_gradiant);

    int test_network(NeuronsNetwork &network, std::vector<TrainingData> &datas);


private:
    void train_network_with_data(NeuronsNetwork &network, const TrainingData &datas, const float &epsilon);
};

#endif // NETWORK_TRAINER_H
