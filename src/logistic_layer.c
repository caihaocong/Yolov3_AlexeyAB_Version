#include "logistic_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_logistic_layer(int batch, int inputs)
{
    fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
//    layer l = {0};
	layer l = { (LAYER_TYPE)0 };  //CH_CONG_20190805
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;
    #ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    #endif
    return l;
}

void forward_logistic_layer(const layer l, network_state net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, LOGISTIC);
    if(net.truth){
        logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_logistic_layer(const layer l, network_state net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU
//CH_CONG_20190807
void forward_logistic_layer_gpu(const layer l, network_state state)
{
	int i, j, k;
	fprintf(stderr, " beginning: forward_logistic_layer_gpu......\n");

	copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
	activate_array_ongpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);

	fprintf(stderr, " free_image(im_truth);......\n");

	if (state.truth)
	{
		logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, state.truth, l.delta_gpu, l.loss_gpu);
		cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
		l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
	}
}

void forward_logistic_layer_gpu_old(const layer l, network_state state)
{
    int i, j, k;

	fprintf(stderr, " beginning: forward_logistic_layer_gpu......\n");

    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);

    cuda_pull_array(l.output_gpu, state.input, l.batch*l.inputs);
    //cuda_pull_array(net.truth_gpu, net.truth, l.batch*l.inputs);

	fprintf(stderr, " beginning: cuda_pull_array......\n");

    image im = make_image(1024, 512, 1);       //CESHI_CONG_20190805
    image im_truth = make_image(1024, 512, 1); //CESHI_CONG_20190805

	fprintf(stderr, "----ceshi---- l.w = %d \n", l.w);
	fprintf(stderr, "----ceshi---- l.h = %d \n", l.h);

    for(i=0; i<l.w*l.h; i++){
      l.delta[i] = 0 - state.input[i];
//      im.data[i] = (float)state.input[i];        //CESHI_CONG_20190805
//      im_truth.data[i] = (float)state.truth[i];  //CESHI_CONG_20190805
    }

//    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);  //CESHI_CONG_20190805
//    save_image(im, "feature_map");		    //CESHI_CONG_20190805
//    save_image(im_truth, "truth");            //CESHI_CONG_20190805
    free_image(im);							  //CESHI_CONG_20190805
    free_image(im_truth);					  //CESHI_CONG_20190805

	fprintf(stderr, " free_image(im_truth);......\n");

    if(state.truth){
        logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, state.truth, l.delta_gpu, l.loss_gpu);
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_logistic_layer_gpu(const layer l, network_state net)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta, 1);
}

#endif
