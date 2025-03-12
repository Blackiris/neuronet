#include "neurons_network_factory.h"

NeuronsNetwork* NeuronsNetworkFactory::create_network(const unsigned int &input_size, const unsigned int &hidden_size, const unsigned int &output_size, const unsigned int &nb_layers) {
    NeuronsNetwork* neurons_network = new NeuronsNetwork();
    unsigned int final_hidden_size = hidden_size;
    if (nb_layers == 1) {
        final_hidden_size = input_size;
    }

    for (unsigned int i=0; i<nb_layers; i++) {
        if (i == nb_layers - 1) {
            neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(output_size, final_hidden_size));
        } else {
            neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(final_hidden_size, input_size));
        }
    }

    return neurons_network;
}

NeuronsNetworkFactory::NeuronsNetworkFactory() {}
