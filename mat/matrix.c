#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 100

Matrix* matrix_create(int row, int col) {
    Matrix *matrix = calloc(1, sizeof(Matrix));
    matrix->rows = row;
    matrix->cols = col;
    matrix->entries = calloc(1, row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        matrix->entries[i] = calloc(1, col * sizeof(double));
    }
    return matrix;
}

void matrix_fill(Matrix* m, int n) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->entries[i][j] = n;
        }
    }
}

void matrix_free(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->entries[i]);
    }
    free(m);
    m = NULL;
}

void matrix_print(Matrix* m) {
    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%1.3f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

Matrix* matrix_copy(Matrix* m) {
    Matrix* matrix = matrix_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix->entries[i][j] = m->entries[i][j];
        }
    }
    return matrix;
}

void matrix_save(Matrix* m, char* file_string) {
    FILE* file = fopen(file_string, "w");
    fprintf(file, "%d\n", m->rows);
    fprintf(file, "%d\n", m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fprintf(file, "%.6f\n", m->entries[i][j]);
        }
    }

    printf("Successfully saved matrix to %s\n", file_string);
    fclose(file);
}

Matrix* matrix_load(char* file_string) {
    FILE* file = fopen(file_string, "r");
    char entry[MAXCHAR];
    fgets(entry, MAXCHAR, file);
    int rows = atoi(entry);
    fgets(entry, MAXCHAR, file);
    int cols = atoi(entry);
    Matrix* m = matrix_create(rows, cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fgets(entry, MAXCHAR, file);
            m->entries[i][j] = strtod(entry, NULL);
        }
    }
    printf("Successfully loaded matrix from %s\n", file_string);
    fclose(file);
    return m;
}

double uniform_distribution(double low, double high) {
    double difference = high - low;
    int scale = 10000;
    int scaled_difference = (int)(difference * scale);
    return low + (1.0 * (rand() % scaled_difference) / scale);
}

void matrix_randomize(Matrix* m, int n) {
    // Random distribution of MIN(-1 / sqrt(n)) and MAX(1 / sqrt(n))
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->entries[i][j] = uniform_distribution(min, max);
        }
    }
}

int matrix_argmax(Matrix* m) {
    // Mx1 matrix
    double max_score = 0;
    int max_idx = 0;
    for (int i = 0; i < m->rows; i++) {
        if (m->entries[i][0] > max_score) {
            max_score = m->entries[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}

Matrix* matrix_flatten(Matrix* m, int axis) {
    // axis:0 -> Column Vector, axis:1 -> Row Vector
    Matrix* matrix;
    if (axis == 0) {
        matrix = matrix_create(m->rows * m->cols, 1);
    }
    else if (axis == 1) {
        matrix = matrix_create(1, m->rows * m->cols);
    }
    else {
        printf("Argument to matrix_flatten must be 0 or 1");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (axis == 0) {
                matrix->entries[i * m->cols + j][0] = m->entries[i][j];
            }
            else if (axis == 1) {
                matrix->entries[0][i * m->cols + j] = m->entries[i][j];
            }
        }
    }
    return matrix;
}
