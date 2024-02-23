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

    int imgs_count = 100;
    Img** imgs = csv_to_imgs("data/mnist_test.csv", imgs_count);

    img_print(imgs[10]);

    NeuralNetwork* net = network_load("testing_network_run_1");
    Matrix* result = network_predict_img(net, imgs[10]);
    matrix_print(result);

    printf("Network predicted %d", matrix_argmax(result));
    
    return 0;
}
