#include "img.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 10000

Img **csv_to_imgs(char* file_string, int number_of_imgs) {
    FILE* f;
    Img** imgs = calloc(1, number_of_imgs * sizeof(Img*));
    char row[MAXCHAR];
    f = fopen(file_string, "r");

    fgets(row, MAXCHAR, f);
    int i = 0;
    while (feof(f) != 1 && i < number_of_imgs) {
        imgs[i] = calloc(1, sizeof(Img));

        int j = -1;
        fgets(row, MAXCHAR, f);
        char* token = strtok(row, ",");
        imgs[i]->img_data = matrix_create(28, 28);
        while (token != NULL) {
            if (j == -1) {
                imgs[i]->label = atoi(token);
            }
            else {
                imgs[i]->img_data->entries[j / 28][j % 28] = atoi(token) / 256.0;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(f);
    return imgs;
}

void img_print(Img* img) {
    matrix_print(img->img_data);
    printf("Img Label: %d\n", img->label);
}

void img_free(Img* img) {
    matrix_free(img->img_data);
    free(img);
    img = NULL;
}

void imgs_free(Img** imgs, int n) {
    for (int i = 0; i < n; i++) {
        img_free(imgs[i]);
    }
    free(imgs);
    imgs = NULL;
}
