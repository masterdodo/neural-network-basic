#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "img/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "mat/matrix.h"
#include "mat/mat_ops.h"


int main() {
    srand(time(NULL));

    int imgs_count = 30000;
    Img** imgs = csv_to_imgs("data/mnist_train.csv", imgs_count);

    NeuralNetwork* net = network_create(784, 300, 10, 0.1);
    network_train_batch_imgs(net, imgs, imgs_count);
    network_save(net, "testing_network_run_1");

    imgs_free(imgs, imgs_count);
    network_free(net);
    
    return 0;
}