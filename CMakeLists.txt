cmake_minimum_required(VERSION 3.16)

project(NeuroNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library( NeuroNetLib STATIC
    network_trainer.h network_trainer.cpp
    training_data.h
    vector.h
    vector_util.h
    std_vector_util.h
    ioutput.h

    neurons_network/ilayer.h neurons_network/ilayer.cpp
    neurons_network/input_layer.h neurons_network/input_layer.cpp
    neurons_network/neurons_layer.h neurons_network/neurons_layer.cpp
    neurons_network/neurons_network.h neurons_network/neurons_network.cpp
    neurons_network/neurons_network_factory.h neurons_network/neurons_network_factory.cpp

    neurons_network/convolution_layer.h neurons_network/convolution_layer.cpp
    neurons_network/ineurons_layer.h neurons_network/ineurons_layer.cpp
    neurons_network/one_to_many_layer.h neurons_network/one_to_many_layer.cpp
    neurons_network/maxpool_layer.h neurons_network/maxpool_layer.cpp
    neurons_network/many_to_many_layer.h neurons_network/many_to_many_layer.cpp
)
target_compile_options(NeuroNetLib PUBLIC -Wall -Wextra -O3 -ffast-math)

SET( APP_EXE NeuroNetTest )


add_executable(${APP_EXE} main.cpp
    cases/case1.h cases/case1.cpp
    cases/case.h
    cases/case_mnist.h cases/case_mnist.cpp
)

TARGET_LINK_LIBRARIES( ${APP_EXE}
                       NeuroNetLib )

include(GNUInstallDirs)
set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/NeuroNet)


install(TARGETS NeuroNetLib
    LIBRARY DESTINATION ${ConfigPackageLocation}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
