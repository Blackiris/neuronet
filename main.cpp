
#include "cases/case1.h"
#include "cases/case_mnist.h"
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "Usage: neuronet casename" << std::endl;
        exit(0);
    }
    std::string case_name(argv[1]);


    if (case_name == "case1") {
        Case1 case1;
        case1.run();
    } else if (case_name == "mnist") {
        CaseMnist caseMnist;
        caseMnist.run();
    }

    return 0;
}
