#include "network_trainer.h"
#include "vector_util.h"
#include <format>

NetworkTrainer::NetworkTrainer() {}


void NetworkTrainer::train_network(NeuronsNetwork &network, const std::vector<std::vector<TrainingData>> &datas_chunks, std::vector<TrainingData> test_datas,
                                   const float &epsilon, const int &nb_iterations, const float &max_gradiant) {
    int iteration(0);

    while(true) {
        for (auto& datas_chunk : datas_chunks) {
            for (auto& data: datas_chunk) {
                iteration++;
                train_network_with_data(network, data, epsilon);

                if (iteration >= nb_iterations) {
                    return;
                }
            }
            int correct = test_network(network, test_datas);
            std::cout << std::format("Iterations {} - {}/{}", iteration, correct, test_datas.size()) << "\n";
            network.apply_new_weights(max_gradiant);

        }
    }
}

unsigned int mapNetworkOutputToRes(const Vector<float> &output) {
    return VectorUtil::find_max_pos<float>(output);
}

bool isResultGood(const Vector<float> &expected, const Vector<float> &actual) {
    return mapNetworkOutputToRes(expected) == mapNetworkOutputToRes(actual);
}

int NetworkTrainer::test_network(NeuronsNetwork& network, std::vector<TrainingData> &datas) {
    unsigned int correct = 0;

    for (unsigned int i=0; i<datas.size(); i++) {
        const TrainingData data = datas[i];
        Vector<float> actual_res = network.compute(data.input);
        if (isResultGood(data.res, actual_res)) {
            correct++;
        }
    }

    return correct;
}


void NetworkTrainer::train_network_with_data(NeuronsNetwork &network, const TrainingData &data, const float &epsilon) {
    auto actual_res = network.compute(data.input);
    Vector<float> dCdZ = (actual_res - data.res) * 2;
    std::vector<Vector<float>> weight_changes;

    //std::cout <<actual_res<< "\n" << data.res<<"\n\n\n";

    for (int i = network.m_layers.size() -1; i>=0; i--) {
        //std::cout <<dCdZ<< "\n";
        std::unique_ptr<NeuronsLayer> &layer = network.m_layers[i];
        ILayer* previous_layer = i > 0 ? (ILayer*)network.m_layers[i-1].get() : &(network.m_input_layer);
        Vector<float> dCdZprime(previous_layer->get_output_size(), 0.f);
        for (unsigned int k=0; k<layer->m_neurons.size(); k++) {

            Neuron &neuron = layer->m_neurons[k];

            //std::cout <<k << "A "<< &layer->m_neurons<< "\n";
            for (int j=0; j<neuron.m_weights.size(); j++) {

                float &weight = neuron.m_weights[j];

                neuron.m_new_weights_delta[j] -= epsilon * dCdZ[k] * neuron.m_deriv_activation_fun(neuron.get_output()) * previous_layer->get_value_at(j);

                dCdZprime[j] += dCdZ[k] * neuron.m_deriv_activation_fun(neuron.get_output()) * weight;
            }
            //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
            //std::cout <<k << "B "<< &layer->m_neurons<< "\n";
        }

        dCdZ = dCdZprime/layer->m_neurons.size();
    }
}
