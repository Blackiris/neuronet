#include "neurons_network_factory.h"
#include "convolution_layer.h"
#include "maxpool_layer.h"
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
                                                           const unsigned int &output_size, const unsigned int &nb_features_map, const unsigned int &nb_features_map2) {
    NeuronsNetwork* neurons_network = new NeuronsNetwork();

    std::vector<INeuronsLayer*> conv_sub_layers;
    conv_sub_layers.reserve(nb_features_map);


    for (unsigned int i=0; i<nb_features_map; i++) {
        ConvolutionLayer* conv_sub_layer = new ConvolutionLayer(input_x, input_y, 1, {0});
        conv_sub_layers.push_back(conv_sub_layer);
    }
    neurons_network->m_layers.push_back(std::make_unique<OneToManyLayer>(conv_sub_layers));
    neurons_network->m_layers.push_back(std::make_unique<MaxpoolLayer>(input_x-2, input_y-2, nb_features_map, 2));

    const std::vector<std::vector<unsigned int>> links_tables = {{0,1,2}, {1,2,3}, {2,3,4},
                                                         {3,4,5}, {0,4,5}, {0,1,5},
                                                         {0,1,2,3}, {1,2,3,4}, {2,3,4,5},
                                                         {0,3,4,5}, {0,1,4,5}, {0,1,2,5},
                                                         {0,1,3,4}, {1,2,4,5}, {0,2,3,5},
                                                         {0,1,2,3,4,5}};
    std::vector<INeuronsLayer*> conv2_sub_layers;
    unsigned int sub_conv2_output_size = 0;
    for (unsigned int i=0; i<nb_features_map2; i++) {
        ConvolutionLayer* conv2_sub_layer = new ConvolutionLayer((input_x-2)/2, (input_y-2)/2, 1, links_tables[i]);
        conv2_sub_layers.push_back(conv2_sub_layer);
        sub_conv2_output_size = conv2_sub_layer->get_output_size();
    }

    neurons_network->m_layers.push_back(std::make_unique<OneToManyLayer>(conv2_sub_layers));
    neurons_network->m_layers.push_back(std::make_unique<MaxpoolLayer>((input_x-2)/2-2, (input_y-2)/2-2, nb_features_map2, 2));

    neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(120, sub_conv2_output_size*nb_features_map2/4));
    neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(84, 120));
    neurons_network->m_layers.push_back(std::make_unique<NeuronsLayer>(output_size, 84));

    return neurons_network;
}
