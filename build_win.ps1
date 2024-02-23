# MATRIX TESTING
gcc -Wall -Wextra -ggdb -o test/matrix_test.exe test/matrix_test.c -lm mat/matrix.c mat/mat_ops.c
# IMG LIB TESTING
gcc -Wall -Wextra -ggdb -o test/img_test.exe test/img_test.c -lm mat/matrix.c mat/mat_ops.c img/img.c

# NEURAL NET TRAINING
gcc -Wall -Wextra -ggdb -o nn_train.exe network_train_test.c -lm mat/matrix.c mat/mat_ops.c img/img.c neural/activations.c neural/nn.c

# NEURAL NET VALIDATION
gcc -Wall -Wextra -ggdb -o nn_val.exe network_val_test.c -lm mat/matrix.c mat/mat_ops.c img/img.c neural/activations.c neural/nn.c

# NEURAL NET BATCH VALIDATION
gcc -Wall -Wextra -ggdb -o nn_val_batch.exe network_val_test_batch.c -lm mat/matrix.c mat/mat_ops.c img/img.c neural/activations.c neural/nn.c
