#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const layer l, network_state net);
void backward_logistic_layer(const layer l, network_state net);

#ifdef GPU
void forward_logistic_layer_gpu(const layer l, network_state net);
void backward_logistic_layer_gpu(const layer l, network_state net);
//CH_CONG_20190807
void forward_logistic_layer_gpu_old(const layer l, network_state state);
#endif

#endif
