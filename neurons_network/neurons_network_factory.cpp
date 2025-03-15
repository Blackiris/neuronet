#include "neurons_network_factory.h"
#include "convolution_layer.h"
#include "neurons_layer.h"
#include "one_to_many_layer.h"

#include <vector>

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

NeuronsNetwork* NeuronsNetworkFactory::create_conv_network(const unsigned int &input_x, const unsigned int &input_y,
                                                           const unsigned int &output_size, const unsigned int &nb_features_map) {
    NeuronsNetwork* neurons_network = new NeuronsNetwork();

    std::vector<INeuronsLayer*> sub_layers;
    sub_layers.reserve(nb_features_map);


    unsigned int conv_output_size;
    for (unsigned int i=0; i<nb_features_map; i++) {
        ConvolutionLayer* conv_sub_layer = new ConvolutionLayer(input_x, input_y, 2);
        sub_layers.push_back(conv_sub_layer);
        conv_output_size = conv_sub_layer->get_output_size();
    }
    std::unique_ptr<OneToManyLayer> conv_layer = std::make_unique<OneToManyLayer>(sub_layers);

    neurons_network->m_layers.push_back(std::move(conv_layer));
    neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(conv_output_size*nb_features_map, output_size));

    return neurons_network;
}
