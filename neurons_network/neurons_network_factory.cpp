#include "neurons_network_factory.h"
#include "convolution_layer.h"
#include "many_to_many_layer.h"
#include "neurons_layer.h"
#include "one_to_many_layer.h"
#include "softmax_layer.h"

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

    std::vector<INeuronsLayer*> conv_sub_layers;
    conv_sub_layers.reserve(nb_features_map);


    unsigned int sub_conv_output_size;
    for (unsigned int i=0; i<nb_features_map; i++) {
        ConvolutionLayer* conv_sub_layer = new ConvolutionLayer(input_x, input_y, 2);
        conv_sub_layers.push_back(conv_sub_layer);
        sub_conv_output_size = conv_sub_layer->get_output_size();
    }
    neurons_network->m_layers.push_back(std::make_unique<OneToManyLayer>(conv_sub_layers));

    std::vector<INeuronsLayer*> softmax_sub_layers;
    for (unsigned int i=0; i<nb_features_map; i++) {
        SoftmaxLayer* softmax_sub_layer = new SoftmaxLayer(input_x-4, input_y-4, 2);
        softmax_sub_layers.push_back(softmax_sub_layer);
    }
    neurons_network->m_layers.push_back(std::make_unique<ManyToManyLayer>(softmax_sub_layers, sub_conv_output_size));

    neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(output_size, sub_conv_output_size*nb_features_map));

    return neurons_network;
}
