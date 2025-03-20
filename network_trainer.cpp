#include "network_trainer.h"
#include "vector_util.h"
#include <format>

NetworkTrainer::NetworkTrainer() {}


void NetworkTrainer::train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, std::vector<TrainingData> test_datas,
                                   const TrainingParams &training_params) {
    int epoch(0);

    while(true) {
        for (auto& datas_chunk : datas_chunks) {
            epoch++;
            double avg_loss = 0;
            for (auto& data: datas_chunk) {
                avg_loss += train_network_with_data(network, data)  / datas_chunk.size();
            }

            network.apply_new_weights(training_params.epsilon/datas_chunk.size(), training_params.max_gradiant);
            int correct = test_network(network, test_datas);
            std::cout << std::format("Epoch {} - {}/{} - loss {}", epoch, correct, test_datas.size(), avg_loss) << "\n";
            if (epoch >= training_params.nb_epochs) {
                return;
            }
        }
    }
}

unsigned int map_network_output_to_res(const Vector<float> &output) {
    return VectorUtil::find_max_pos<float>(output);
}

bool is_prediction_good(const Vector<float> &expected, const Vector<float> &actual) {
    return map_network_output_to_res(expected) == map_network_output_to_res(actual);
}

int NetworkTrainer::test_network(NeuronsNetwork& network, std::vector<TrainingData> &datas) {
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


double NetworkTrainer::train_network_with_data(NeuronsNetwork &network, const TrainingData &data) {
    auto actual_res = network.compute(data.input);
    Vector<float> error = data.res - actual_res;
    Vector<float> dCdZ = error;
    double loss = dCdZ.length();
    std::vector<Vector<float>> weight_changes;

    //std::cout <<actual_res<< "\n" << data.res<<"\n\n\n";

    for (int i = network.m_layers.size() -1; i>=0; i--) {

        std::unique_ptr<INeuronsLayer> &layer = network.m_layers[i];
        ILayer* previous_layer = i > 0 ? (ILayer*)network.m_layers[i-1].get() : &(network.m_input_layer);

        Vector<float> output = previous_layer->get_output();

        Vector<float> dCdZprime = layer->adapt_gradient(output, dCdZ);
        //std::cout << "dcdz "<<i<<": " << dCdZ<< " (s:"<< dCdZ.size()<< " - l:"<< dCdZ.length()<< ") "<<")\ndcdzprime "<<i<<": " << dCdZprime<< "(s:"<<dCdZprime.size() << " - l:"<<dCdZprime.length() <<")\n\n";
        dCdZ = dCdZprime;
    }
    return loss;
}
