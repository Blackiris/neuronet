#ifndef NETWORK_TRAINER_H
#define NETWORK_TRAINER_H

#include <vector>
#include "training_data.h"
#include "neurons_network/neurons_network.h"

struct TrainingParams {
    float epsilon;
    int nb_epochs;
    float max_gradiant;
};

class NetworkTrainer
{
public:
    NetworkTrainer();

    void train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, std::vector<TrainingData> test_datas,
                       const TrainingParams &training_params);

    int test_network(NeuronsNetwork &network, std::vector<TrainingData> &datas);


private:
    float train_network_with_data(NeuronsNetwork &network, const TrainingData &datas, const float &epsilon);
};

#endif // NETWORK_TRAINER_H
