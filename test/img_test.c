#include "../mat/matrix.h"
#include "../mat/mat_ops.h"
#include "../img/img.h"

int main() {
    Img **imgs = csv_to_imgs("data/mnist_train.csv", 10);
    img_print(imgs[0]);

    return 0;
}
