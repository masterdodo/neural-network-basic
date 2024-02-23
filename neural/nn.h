#pragma once

#include "../mat/matrix.h"
#include "../img/img.h"

typedef struct
{
    int input;            // Number of input neurons
    int hidden;           // Number of hidden neurons
    int output;           // Number of output neurons
    double learning_rate; // LR - hyper param
    Matrix *hidden_weights;
    Matrix *output_weights;
} NeuralNetwork;

NeuralNetwork *network_create(int input, int hidden, int output, double lr);
void network_train(NeuralNetwork *net, Matrix *input_data, Matrix *output_data);
void network_train_batch_imgs(NeuralNetwork *net, Img **imgs, int batch_size);
Matrix *network_predict_img(NeuralNetwork *net, Img *img);
double network_predict_imgs(NeuralNetwork *net, Img **imgs, int n);
Matrix *network_predict(NeuralNetwork *net, Matrix *input_data);
void network_save(NeuralNetwork *net, char *file_string);
NeuralNetwork *network_load(char *file_string);
void network_print(NeuralNetwork *net);
void network_free(NeuralNetwork *net);
