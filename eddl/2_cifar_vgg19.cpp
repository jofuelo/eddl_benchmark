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

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////

layer defblock(layer l, bool bn, int nf, int reps){
    for(int i = 0; i < reps; i++){
      l = GlorotUniform(Conv(l, nf, {3, 3}));
      if(bn) l = BatchNormalization(l, 0.99, 0.001, true, "");
      l = ReLu(l);
    }
    l = MaxPool(l, {2, 2}, {2, 2}, "valid"); 
    return l;
}

int main(int argc, char **argv) {
  // download CIFAR data
  download_cifar10();

    // Settings
    int epochs = 50;
    int batch_size = 100;
    int num_classes = 10;

    bool bn = true;

    // Define network
    layer in=Input({3, 32, 32});
    layer l = in;  // Aux var

    l = defblock(l, bn, 64, 2);
    l = defblock(l, bn, 128, 2);
    l = defblock(l, bn, 256, 4);
    l = defblock(l, bn, 512, 4);
    l = defblock(l, bn, 512, 4);

    l = Flatten(l);
    for(int i = 0; i < 2; i++){
      l = GlorotUniform(Dense(l, 4096));
      if(bn) l = BatchNormalization(l, 0.99, 0.001, true, "");;
      l = ReLu(l);
    }
    layer out = Softmax(GlorotUniform(Dense(l, num_classes)));

    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
		adam(0.00001), // Optimizer
        {"soft_cross_entropy"}, // Losses
        {"categorical_accuracy"}, // Metrics
        CS_GPU({1}) // one GPU
    );

    // View model
    summary(net);

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
