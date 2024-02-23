#include "../mat/matrix.h"
#include "../mat/mat_ops.h"

#include <stdio.h>

int main() {
    printf("\nMatrix test: add\n");
    Matrix *m1 = matrix_create(2, 2);
    Matrix *m2 = matrix_create(2, 2);
    m1->entries[0][0] = 1;
    matrix_fill(m1, 2);
    m2->entries[1][1] = 5;
    Matrix *m3 = add(m1, m2);
    matrix_print(m3);

    printf("\nMatrix test: dot\n");
    Matrix *m4 = matrix_create(2, 3);
    Matrix *m5 = matrix_create(3, 2);
    m4->entries[0][0] = 3;
    m4->entries[0][1] = 2;
    m4->entries[0][2] = 4;
    m4->entries[1][0] = 9;
    m4->entries[1][1] = 7;
    m4->entries[1][2] = 6;
    m5->entries[0][0] = 1;
    m5->entries[0][1] = 5;
    m5->entries[1][0] = 3;
    m5->entries[1][1] = 9;
    m5->entries[2][0] = 7;
    m5->entries[2][1] = 4;
    matrix_print(m4);
    matrix_print(m5);
    Matrix *m6 = dot(m4, m5);
    matrix_print(m6);

    printf("\nMatrix test: transpose\n");
    Matrix *m7 = matrix_create(2, 3);
    matrix_fill(m7, 0);
    Matrix *m8 = transpose(m7);
    matrix_print(m7);
    matrix_print(m8);

    Matrix *m9 = matrix_create(10, 1);
    m9->entries[0][0] = 0;
    m9->entries[1][0] = 0;
    m9->entries[2][0] = 0;
    m9->entries[3][0] = 0;
    m9->entries[4][0] = 1;
    m9->entries[5][0] = 0;
    m9->entries[6][0] = 0;
    m9->entries[7][0] = 0;
    m9->entries[8][0] = 0;
    m9->entries[9][0] = 0;
    printf("Matrix argmax: %d", matrix_argmax(m9));

    return 0;
}