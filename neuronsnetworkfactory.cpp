#include "neuronsnetworkfactory.h"

NeuronsNetwork* NeuronsNetworkFactory::createNetwork(const unsigned int &input_size, const unsigned int &output_size, const unsigned int &nb_layers) {
    NeuronsNetwork* neurons_network = new NeuronsNetwork();

    for (unsigned int i=0; i<nb_layers; i++) {
        if (i == nb_layers - 1) {
            neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(output_size, input_size));
        } else {
            neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(input_size, input_size));
        }
    }

    return neurons_network;
}

NeuronsNetworkFactory::NeuronsNetworkFactory() {}
