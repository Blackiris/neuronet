#include "network_trainer.h"
#include <format>

NetworkTrainer::NetworkTrainer() {}


void NetworkTrainer::train_network(NeuronsNetwork &network, const std::vector<TrainingData> &datas, const float &epsilon, const int &nb_iterations) {

    int iteration(0);
    while (iteration < nb_iterations) {
        train_network_with_data(network, datas[iteration % datas.size()], epsilon);
        iteration++;
        std::cout <<"Iteration "<<iteration<< "\n";
    }
}


void NetworkTrainer::train_network_with_data(NeuronsNetwork &network, const TrainingData &data, const float &epsilon) {
    auto actual_res = network.compute(data.input);
    Vector<float> dCdZ = (actual_res - data.res) * 2;

    std::cout <<actual_res<< "\n" << data.res<<"\n\n\n";

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

                weight -= epsilon * dCdZ[k] * neuron.m_deriv_activation_fun(neuron.get_output()) * previous_layer->get_value_at(j);

                dCdZprime[j] += dCdZ[k] * neuron.m_deriv_activation_fun(neuron.get_output()) * weight;
            }
            //std::cout << std::format("Neurone {} - {}", i, k) << " dCdz " << dCdZ[k] << " Weight: " << neuron.m_weights << "\n";
            //std::cout <<k << "B "<< &layer->m_neurons<< "\n";
        }

        dCdZ = dCdZprime/layer->m_neurons.size();
    }
}
