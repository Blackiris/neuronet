#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include "vector.h"

struct TrainingData {
    Vector<float> input;
    Vector<float> res;
};

struct TrainingParams {
    float epsilon;
    float epsilon_bias;
    int nb_epochs;
    float clip_gradiant_threshold = 0;
    float adam_decay_rate_momentum = 0;
    float adam_decay_rate_squared = 0;
    int current_epoch = 0;
};

#endif // TRAININGDATA_H
