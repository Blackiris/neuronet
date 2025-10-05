cmake_minimum_required(VERSION 3.16)

project(NeuroNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET( APP_EXE NeuroNetTest )
SET( LIB_NAME NeuroNetLib )

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_subdirectory(neurons_network)



add_executable(${APP_EXE} main.cpp
    cases/case1.h cases/case1.cpp
    cases/case.h
    cases/case_mnist.h cases/case_mnist.cpp
)

target_link_libraries(${APP_EXE} ${LIB_NAME})

include(GNUInstallDirs)
set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/NeuroNet)

set_target_properties(${LIB_NAME} PROPERTIES PUBLIC_HEADER network_trainer.h)
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/neuronetlib
)
