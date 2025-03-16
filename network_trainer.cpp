#include "network_trainer.h"
#include "vector_util.h"
#include <format>

NetworkTrainer::NetworkTrainer() {}


void NetworkTrainer::train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, std::vector<TrainingData> test_datas,
                                   const TrainingParams &training_params) {
    int iteration(0);

    while(true) {
        for (auto& datas_chunk : datas_chunks) {
            iteration++;
            float avg_error_length = 0;
            for (auto& data: datas_chunk) {
                avg_error_length += train_network_with_data(network, data, training_params.epsilon);
            }

            network.apply_new_weights(training_params.max_gradiant);
            int correct = test_network(network, test_datas);
            std::cout << std::format("Epoch {} - {}/{} - avg error length {}", iteration, correct, test_datas.size(), avg_error_length/datas_chunks.size()) << "\n";
            if (iteration >= training_params.nb_epochs) {
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


float NetworkTrainer::train_network_with_data(NeuronsNetwork &network, const TrainingData &data, const float &epsilon) {
    auto actual_res = network.compute(data.input);
    Vector<float> error = actual_res - data.res;
    float error_length = error.length();
    Vector<float> dCdZ = error * 2;
    std::vector<Vector<float>> weight_changes;

    //std::cout <<actual_res<< "\n" << data.res<<"\n\n\n";

    for (int i = network.m_layers.size() -1; i>=0; i--) {
        //std::cout <<dCdZ<< "\n";
        std::unique_ptr<INeuronsLayer> &layer = network.m_layers[i];
        ILayer* previous_layer = i > 0 ? (ILayer*)network.m_layers[i-1].get() : &(network.m_input_layer);

        Vector<float> output = previous_layer->get_output();

        Vector<float> dCdZprime = layer->adapt_gradient(output, dCdZ, epsilon);


        dCdZ = dCdZprime/layer->get_output_size();
    }
    return error_length;
}
