#ifndef NEURONSNETWORKFACTORY_H
#define NEURONSNETWORKFACTORY_H

#include "neurons_network.h"

class NeuronsNetworkFactory
{
public:
    static NeuronsNetwork* createNetwork();
private:
    NeuronsNetworkFactory();
};

#endif // NEURONSNETWORKFACTORY_H
