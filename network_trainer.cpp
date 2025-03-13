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
            for (auto& data: datas_chunk) {
                train_network_with_data(network, data, training_params.epsilon);
            }

            network.apply_new_weights(training_params.max_gradiant);
            int correct = test_network(network, test_datas);
            std::cout << std::format("Epoch {} - {}/{}", iteration, correct, test_datas.size()) << "\n";
            if (iteration >= training_params.nb_epochs) {
                return;
            }
        }
    }
}

unsigned int map_network_output_to_res(const Vector<float> &output) {
    return VectorUtil::find_max_pos<float>(output);
}

bool isResultGood(const Vector<float> &expected, const Vector<float> &actual) {
    return map_network_output_to_res(expected) == map_network_output_to_res(actual);
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

            neuron.adapt_gradient(*previous_layer, dCdZ[k], epsilon, dCdZprime);

            //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";

        }

        dCdZ = dCdZprime/layer->m_neurons.size();
    }
}
