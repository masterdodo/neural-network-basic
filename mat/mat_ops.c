#include "mat_ops.h"
#include <stdlib.h>
#include <stdio.h>

int check_dimensions(Matrix* m1, Matrix* m2) {
    if (m1->rows == m2->rows && m1->cols == m2->cols) {
        return 1;
    }
    return 0;
}

Matrix* multiply(Matrix *m1, Matrix *m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
    Matrix* matrix = matrix_create(m1->rows, m1->cols);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            matrix->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
        }
    }
    return matrix;
}

Matrix* add(Matrix *m1, Matrix *m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
    Matrix* matrix = matrix_create(m1->rows, m1->cols);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            matrix->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
        }
    }
    return matrix;
}

Matrix* subtract(Matrix *m1, Matrix *m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
    Matrix* matrix = matrix_create(m1->rows, m1->cols);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            matrix->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
        }
    }
    return matrix;
}

Matrix* apply(double (*func)(double), Matrix* m) {
    Matrix *matrix = matrix_copy(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix->entries[i][j] = (*func)(m->entries[i][j]);
        }
    }
    return matrix;
}

Matrix* scale(double n, Matrix *m) {
    Matrix* matrix = matrix_copy(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix->entries[i][j] *= n;
        }
    }
    return matrix;
}

Matrix* addScalar(double n, Matrix *m) {
    Matrix* matrix = matrix_copy(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix->entries[i][j] += n;
        }
    }
    return matrix;
}

Matrix* transpose(Matrix *m) {
    Matrix* matrix = matrix_create(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            matrix->entries[j][i] = m->entries[i][j];
        }
    }
    return matrix;
}

Matrix* dot(Matrix *m1, Matrix *m2) {
    if (m1->cols != m2->rows) {
        printf("Dimension mismatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }
    Matrix *matrix = matrix_create(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            double sum = 0;
            for (int k = 0; k < m2->rows; k++) {
                sum += m1->entries[i][k] * m2->entries[k][j];
            }
            matrix->entries[i][j] = sum;
        }
    }
    return matrix;
}
