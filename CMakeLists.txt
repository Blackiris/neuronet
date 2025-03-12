cmake_minimum_required(VERSION 3.16)

project(NeuroNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(NeuroNet main.cpp



    network_trainer.h network_trainer.cpp
    training_data.h
    vector.h vector.cpp


    cases/case1.h cases/case1.cpp
    cases/Case.h
    cases/case_mnist.h cases/case_mnist.cpp

    vector_util.h
    vector_util.cpp
    std_vector_util.h

    ioutput.h
    neurons_network/dummy.h neurons_network/dummy.cpp
    neurons_network/neuron.h neurons_network/neuron.cpp
    neurons_network/ilayer.h neurons_network/ilayer.cpp
    neurons_network/input_layer.h neurons_network/input_layer.cpp
    neurons_network/neurons_layer.h neurons_network/neurons_layer.cpp
    neurons_network/neurons_network.h neurons_network/neurons_network.cpp
    neurons_network/neuronsnetworkfactory.h neurons_network/neuronsnetworkfactory.cpp

)

include(GNUInstallDirs)
install(TARGETS NeuroNet
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
