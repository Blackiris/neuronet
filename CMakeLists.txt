cmake_minimum_required(VERSION 3.16)

project(NeuroNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(NeuroNet main.cpp
    ioutput.h ioutput.cpp
    neuron.h neuron.cpp
    neurons_network.h
    neurons_network.cpp
    network_trainer.h network_trainer.cpp
    training_data.h
    vector.h vector.cpp
    ilayer.h
    input_layer.h input_layer.cpp
    neurons_layer.h neurons_layer.cpp
    ilayer.cpp
)

include(GNUInstallDirs)
install(TARGETS NeuroNet
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
