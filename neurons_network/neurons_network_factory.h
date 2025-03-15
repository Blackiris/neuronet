#ifndef NEURONS_NETWORK_FACTORY_H
#define NEURONS_NETWORK_FACTORY_H

#include "neurons_network.h"

class NeuronsNetworkFactory
{
public:
    static NeuronsNetwork* create_network(const unsigned int &input_size, const unsigned int &hidden_size, const unsigned int &output_size, const unsigned int &nb_layers);
    static NeuronsNetwork* create_conv_network(const unsigned int &input_x, const unsigned int &input_y,
                                               const unsigned int &output_size, const unsigned int &nb_features_map);
private:
    NeuronsNetworkFactory();
};

#endif // NEURONS_NETWORK_FACTORY_H
