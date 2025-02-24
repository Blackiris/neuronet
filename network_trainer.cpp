#include "network_trainer.h"


NetworkTrainer::NetworkTrainer() {}


void NetworkTrainer::train_network(NeuronsNetwork &network, const std::vector<TrainingData> &datas, const float &epsilon, const int &nb_iterations) {

    int iteration(0);
    while (iteration < nb_iterations) {
        train_network_with_data(network, datas[iteration % datas.size()], epsilon);
        iteration++;
    }
}


void NetworkTrainer::train_network_with_data(NeuronsNetwork &network, const TrainingData &data, const float &epsilon) {
    auto actual_res = network.compute(data.input);
    Vector<float> dCdZ = (actual_res - data.res) * 2;
    Vector<float> dCdZprime(dCdZ.size(), 0.f);
//std::cout <<dCdZ<< "\n";
    for (int i = network.m_layers.size() -1; i>=0; i--) {
        std::unique_ptr<NeuronsLayer> &layer = network.m_layers[i];
        ILayer* previous_layer = i > 0 ? (ILayer*)network.m_layers[i-1].get() : &(network.m_input_layer);
        for (int k=0; k<layer->m_neurons.size(); k++) {
            Neuron &neuron = layer->m_neurons[k];
            for (int j=0; j<neuron.m_weights.size(); j++) {
                float &weight = neuron.m_weights[j];
                weight -= epsilon * dCdZ[k] * previous_layer->get_value_at(j);
                dCdZprime[k] += dCdZ[k] * weight;
            }
            //std::cout << "Neurone " << k << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
        }

        dCdZ = dCdZprime;
    }

    Vector<float> dCdWi = (actual_res - data.res) * 2;
}
