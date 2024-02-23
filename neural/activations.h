#pragma once

#include "../mat/matrix.h"

double sigmoid(double input);
Matrix *sigmoidPrime(Matrix *m);
Matrix *softmax(Matrix *m);
