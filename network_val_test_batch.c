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

    int imgs_count = 3000;
    Img** imgs = csv_to_imgs("data/mnist_test.csv", imgs_count);

    NeuralNetwork* net = network_load("testing_network_run_1");
    double score = network_predict_imgs(net, imgs, 1000);

    printf("Accuracy: %1.5f", score);
    
    return 0;
}
