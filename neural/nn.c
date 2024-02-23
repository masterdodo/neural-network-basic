#include "nn.h"

#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../mat/mat_ops.h"
#include "activations.h"

#define MAXCHAR 1000

NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net = calloc(1, sizeof(NeuralNetwork));
    net->input = input;
    net->hidden = hidden;
    net->output = output;
    net->learning_rate = lr;

    Matrix* hidden_layer = matrix_create(hidden, input);
    Matrix* output_layer = matrix_create(output, hidden);
    matrix_randomize(hidden_layer, hidden);
    matrix_randomize(output_layer, output);
    net->hidden_weights = hidden_layer;
    net->output_weights = output_layer;

    return net;
}

void network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
    // 1. Feed Forward
    Matrix* hidden_inputs = dot(net->hidden_weights, input); // Use dot for hidden layer input
    Matrix* hidden_outputs = apply(sigmoid, hidden_inputs); // Output from the hidden layer (sigmoid activation curve)
    Matrix* final_inputs = dot(net->output_weights, hidden_outputs); // Use dot for output layer input
    Matrix* final_outputs = apply(sigmoid, final_inputs); // Final output of the net by same sigmoid activation function

    // 2. Compute Errors
    Matrix* output_errors = subtract(output, final_outputs); // We subtract the final outputs from the expected output
    Matrix* hidden_errors = dot(transpose(net->output_weights), output_errors);

    // 3. Backpropagation (confusion overload)

    // Output Weights
    Matrix* sigmoid_primed_matrix = sigmoidPrime(final_outputs);
    Matrix* multiplied_matrix = multiply(output_errors, sigmoid_primed_matrix);
    Matrix* transposed_matrix = transpose(hidden_outputs);
    Matrix* dot_matrix = dot(multiplied_matrix, transposed_matrix);
    Matrix* scaled_matrix = scale(net->learning_rate, dot_matrix);
    Matrix* added_matrix = add(net->output_weights, scaled_matrix);

    matrix_free(net->output_weights);
    net->output_weights = added_matrix;
    // Free matrices
    matrix_free(sigmoid_primed_matrix);
    matrix_free(multiplied_matrix);
    matrix_free(transposed_matrix);
    matrix_free(dot_matrix);
    matrix_free(scaled_matrix);

    // Hidden Weights
    sigmoid_primed_matrix = sigmoidPrime(hidden_outputs);
    multiplied_matrix = multiply(hidden_errors, sigmoid_primed_matrix);
    transposed_matrix = transpose(input);
    dot_matrix = dot(multiplied_matrix, transposed_matrix);
    scaled_matrix = scale(net->learning_rate, dot_matrix);
    added_matrix = add(net->hidden_weights, scaled_matrix);

    matrix_free(net->hidden_weights);
    net->hidden_weights = added_matrix;
    // Free matrices
    matrix_free(sigmoid_primed_matrix);
    matrix_free(multiplied_matrix);
    matrix_free(transposed_matrix);
    matrix_free(dot_matrix);
    matrix_free(scaled_matrix);
    // Free All
    matrix_free(hidden_inputs);
    matrix_free(hidden_outputs);
    matrix_free(final_inputs);
    matrix_free(final_outputs);
    matrix_free(output_errors);
    matrix_free(hidden_errors);
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        if (i % 100 == 0) {
            printf("Img No. %d\n", i);
        }

        Img* curr_img = imgs[i];
        Matrix* img_data = matrix_flatten(curr_img->img_data, 0); // Column Vector
        Matrix* output = matrix_create(10, 1); // Output Layer
        output->entries[curr_img->label][0] = 1; // The expected result
        network_train(net, img_data, output);

        matrix_free(output);
        matrix_free(img_data);
    }
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
    Matrix* hidden_inputs = dot(net->hidden_weights, input_data);
    Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
    Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
    Matrix* final_outputs = apply(sigmoid, final_inputs);

    Matrix* result = softmax(final_outputs);

    matrix_free(hidden_inputs);
    matrix_free(hidden_outputs);
    matrix_free(final_inputs);
    matrix_free(final_outputs);

    return result;
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
    Matrix* img_data = matrix_flatten(img->img_data, 0); // Column Vector
    Matrix* result = network_predict(net, img_data);
    matrix_free(img_data);
    
    return result;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        Matrix* prediction = network_predict_img(net, imgs[i]);
        
        if (matrix_argmax(prediction) == imgs[i]->label) {
            correct++;
        }
        matrix_free(prediction);
    }

    return (1.0 * correct) / n;
}

void network_save(NeuralNetwork* net, char* file_string) {
    mkdir(file_string);
    chdir(file_string);

    FILE* descriptor = fopen("descriptor", "w");
    fprintf(descriptor, "%d\n", net->input);
    fprintf(descriptor, "%d\n", net->hidden);
    fprintf(descriptor, "%d\n", net->output);
    fclose(descriptor);

    matrix_save(net->hidden_weights, "hidden");
    matrix_save(net->output_weights, "output");
    printf("Successfully written to '%s'\n", file_string);

    chdir("-"); // Back to original dir
}

NeuralNetwork* network_load(char* file_string) {
    NeuralNetwork* net = calloc(1, sizeof(NeuralNetwork));
    char entry[MAXCHAR];
    chdir(file_string);

    FILE* descriptor = fopen("descriptor", "r");
    
    fgets(entry, MAXCHAR, descriptor);
    net->input = atoi(entry);
    fgets(entry, MAXCHAR, descriptor);
    net->hidden = atoi(entry);
    fgets(entry, MAXCHAR, descriptor);
    net->output = atoi(entry);

    fclose(descriptor);

    net->hidden_weights = matrix_load("hidden");
    net->output_weights = matrix_load("output");
    printf("Successfully loaded network from '%s'\n", file_string);

    chdir("-"); // Back to original dir

    return net;
}

void network_print(NeuralNetwork* net) {
    printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hidden_weights);
	matrix_free(net->output_weights);
	free(net);
	net = NULL;
}
