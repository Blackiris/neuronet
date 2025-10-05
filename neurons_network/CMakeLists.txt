add_library(${LIB_NAME} STATIC
    network_trainer.h network_trainer.cpp

    ilayer.h ilayer.cpp
    input_layer.h input_layer.cpp
    neurons_layer.h neurons_layer.cpp
    neurons_network.h neurons_network.cpp
    neurons_network_factory.h neurons_network_factory.cpp

    convolution_layer.h convolution_layer.cpp
    ineurons_layer.h ineurons_layer.cpp
    one_to_many_layer.h one_to_many_layer.cpp
    maxpool_layer.h maxpool_layer.cpp
    many_to_many_layer.h many_to_many_layer.cpp
)
target_compile_options(${LIB_NAME} PUBLIC -O3 -ffast-math)
