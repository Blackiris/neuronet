#include "neuronsnetworkfactory.h"

NeuronsNetwork* NeuronsNetworkFactory::createNetwork() {
    NeuronsNetwork* neurons_network = new NeuronsNetwork();
    std::vector<std::unique_ptr<NeuronsLayer>> m_layers;
    std::unique_ptr<NeuronsLayer> neurons_layer(new NeuronsLayer(10, 10));
    std::unique_ptr<NeuronsLayer> neurons_layer2(new NeuronsLayer(10, 10));
    neurons_network->m_layers.push_back(std::move(neurons_layer));
    neurons_network->m_layers.push_back(std::move(neurons_layer2));

    return neurons_network;
}

NeuronsNetworkFactory::NeuronsNetworkFactory() {}
