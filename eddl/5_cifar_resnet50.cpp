/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

layer resnet_block(layer l0, int nf, bool bn, int reps, bool downsample){
  layer l1;
  for(int i = 0; i < reps; i++){
    int stri = (downsample and i==0) ? 2 : 1;

    l1 = GlorotUniform(Conv(l0, nf, {1, 1}, {stri,stri}, "same", false));
    if(bn) l1 = BatchNormalization(l1, 0.99, 0.001, true, "");
    l1 = ReLu(l1);
    l1 = GlorotUniform(Conv(l1, nf, {3, 3}, {1,1}, "same", false));
    if(bn) l1 = BatchNormalization(l1, 0.99, 0.001, true, "");
    l1 = ReLu(l1);
    l1 = GlorotUniform(Conv(l1, nf*4, {1, 1}, {1,1}, "same", false));
    if(bn) l1 = BatchNormalization(l1, 0.99, 0.001, true, "");

    if(i==0)
        l0 = GlorotUniform(Conv(l0, nf*4, {1,1}, {stri,stri}, "same", false));

    l0=Add({l0,l1});
    l0 = ReLu(l0);
  }
  return l0;
}

int main(int argc, char **argv) {
  // download CIFAR data
  download_cifar10();

    // Settings
    int epochs = 50;
    int batch_size = 50;
    int num_classes = 10;

    bool bn = false;


    // Define network
    layer in=Input({3, 32, 32});
    layer l = in;  // Aux var
  	l = GlorotUniform(Conv(l, 64, {7, 7}, {2,2}, "same", false));
    l = MaxPool(l, {2, 2}, {2, 2}, "valid");
    l = resnet_block(l, 64, bn, 3, false);
    l = resnet_block(l, 128, bn, 4, true);
    l = resnet_block(l, 256, bn, 6, true);
    l = resnet_block(l, 512, bn, 3, true);
    l = GlobalAveragePool(l);
    l = Flatten(l);
    layer out = Softmax(GlorotUniform(Dense(l, num_classes)));

    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
		adam(0.0001), // Optimizer
        {"soft_cross_entropy"}, // Losses
        {"categorical_accuracy"}, // Metrics
        CS_GPU({1}) // one GPU
    );

    // View model
    //summary(net);

	// Load and preprocess training data
	Tensor* x_train = Tensor::load("cifar_trX.bin");
	Tensor* y_train = Tensor::load("cifar_trY.bin");
	x_train->div_(255.0f);

	// Load and preprocess test data
	Tensor* x_test = Tensor::load("cifar_tsX.bin");
	Tensor* y_test = Tensor::load("cifar_tsY.bin");
	x_test->div_(255.0f);

	for(int i=0;i<epochs;i++) {
    	std::cout << i << std::endl;
    	// training, list of input and output tensors, batch, epochs
    	fit(net,{x_train},{y_train},batch_size, 1);

    	// Evaluate train
    	std::cout << "Evaluate test:" << std::endl;
    	evaluate(net,{x_test},{y_test});
	}

}
