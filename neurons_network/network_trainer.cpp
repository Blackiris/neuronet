#include "network_trainer.h"
#include "vector_util.h"
#include <format>

NetworkTrainer::NetworkTrainer() {}


void NetworkTrainer::train_network(NeuronsNetwork &network,
                                   const std::vector<std::vector<TrainingData>> &datas_chunks,
                                   const std::vector<TrainingData> &test_datas,
                                   const TrainingParams &training_params) {
    int epoch(0);

    while(true) {
        for (const auto& datas_chunk : datas_chunks) {
            const unsigned int chunk_size = datas_chunk.size();
            epoch++;

            const auto loss_fold = [&](double total, const TrainingData &data) {
                return total + train_network_with_data(network, data);
            };
            const double avg_loss = std::accumulate(datas_chunk.begin(), datas_chunk.end(), 0.f, loss_fold)
                                    / chunk_size;

            TrainingParams training_params_local = training_params;
            training_params_local.epsilon /= chunk_size;
            training_params_local.epsilon_bias /= chunk_size;
            training_params_local.current_epoch = epoch;
            network.apply_new_weights(training_params_local);
            int correct_chunk = test_network(network, datas_chunk);
            int correct_test = test_network(network, test_datas);
            std::cout << "Epoch " << epoch << " - acc_train " << correct_chunk << "/" << chunk_size << " - acc_test "<< correct_test <<"/" << test_datas.size() << " - loss "<< avg_loss << "\n";
            if (epoch >= training_params.nb_epochs) {
                return;
            }
        }
    }
}

unsigned int NetworkTrainer::map_network_output_to_res(const Vector<float> &output) {
    return VectorUtil::find_max_pos<float>(output);
}

bool NetworkTrainer::is_prediction_good(const Vector<float> &expected, const Vector<float> &actual) {
    return map_network_output_to_res(expected) == map_network_output_to_res(actual);
}

int NetworkTrainer::test_network(NeuronsNetwork& network, const std::vector<TrainingData> &datas) {
    unsigned int correct = 0;

    for (unsigned int i=0; i<datas.size(); i++) {
        const TrainingData data = datas[i];
        Vector<float> actual_res = network.compute(data.input);
        if (is_prediction_good(data.res, actual_res)) {
            correct++;
        }
    }

    return correct;
}


double NetworkTrainer::train_network_with_data(NeuronsNetwork &network, const TrainingData &datas) {
    auto actual_res = network.compute(datas.input);
    Vector<float> error = datas.res - actual_res;
    Vector<float> dCdZ = error;
    double loss = dCdZ.length();

    //std::cout <<actual_res<< "\n" << data.res<<"\n\n\n";

    for (int i = network.m_layers.size() -1; i>=0; i--) {

        std::unique_ptr<INeuronsLayer> &layer = network.m_layers[i];
        const ILayer* previous_layer = i > 0 ? static_cast<ILayer*>(network.m_layers[i-1].get()) : &(network.m_input_layer);

        const Vector<float> &prev_output = previous_layer->get_output();

        Vector<float> dCdZprime(prev_output.size(), 0);
        layer->adapt_gradient(prev_output, dCdZ, dCdZprime, 0);
        //std::cout << "dcdz "<<i<<": " << dCdZ<< " (s:"<< dCdZ.size()<< " - l:"<< dCdZ.length()<< ") "<<")\ndcdzprime "<<i<<": " << dCdZprime<< "(s:"<<dCdZprime.size() << " - l:"<<dCdZprime.length() <<")\n\n";
        dCdZ = dCdZprime;
    }
    return loss;
}
