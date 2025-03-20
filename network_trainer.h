#ifndef NETWORK_TRAINER_H
#define NETWORK_TRAINER_H

#include <vector>
#include "training_data.h"
#include "neurons_network/neurons_network.h"

struct TrainingParams {
    float epsilon;
    float epsilon_bias;
    int nb_epochs;
    float clip_gradiant_threshold = 0;
};

class NetworkTrainer
{
public:
    NetworkTrainer();

    void train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, std::vector<TrainingData> test_datas,
                       const TrainingParams &training_params);

    int test_network(NeuronsNetwork &network, std::vector<TrainingData> &datas);


private:
    double train_network_with_data(NeuronsNetwork &network, const TrainingData &datas);
    bool is_prediction_good(const Vector<float> &expected, const Vector<float> &actual);
    unsigned int map_network_output_to_res(const Vector<float> &output);
};

#endif // NETWORK_TRAINER_H
