#ifndef NEURONSNETWORKFACTORY_H
#define NEURONSNETWORKFACTORY_H

#include "neurons_network.h"

class NeuronsNetworkFactory
{
public:
    static NeuronsNetwork* createNetwork(const unsigned int &input_size, const unsigned int &hidden_size, const unsigned int &output_size, const unsigned int &nb_layers);
private:
    NeuronsNetworkFactory();
};

#endif // NEURONSNETWORKFACTORY_H
